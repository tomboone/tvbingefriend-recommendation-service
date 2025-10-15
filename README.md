# TV BingeFriend Recommendation Service

Content-based TV show recommendation system using hybrid similarity scoring.

## Architecture Overview

### Components

1. **Azure Functions** - API layer for serving recommendations from MySQL
2. **Azure Container Instance** - Data pipeline container for processing shows and computing similarities
3. **Azure Logic App** - Scheduler that triggers the container instance weekly
4. **Azure Blob Storage** - Storage for feature matrices only (~213 MB)
5. **MySQL Database** - Primary data store for show metadata and pre-computed top-N recommendations

### Data Flow

```
┌─────────────────────────────────────┐
│  Azure Logic App                    │
│  (Recurring Schedule: Weekly)       │
│                                     │
│  Triggers container instance        │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  Azure Container Instance                                │
│                                                          │
│  1. Fetch shows from API                                 │
│  2. Compute features (genre, text, metadata)             │
│  3. Upload features to Blob Storage                      │
│  4. Run database migrations                              │
│  5. Populate MySQL Database:                             │
│     - Sync show metadata                                 │
│     - Compute similarities (in-memory, per show)         │
│     - Store top N recommendations incrementally          │
└────────────────────┬─────────────────┬───────────────────┘
                     │                 │
                     │                 │
        ┌────────────▼──────┐    ┌─────▼───────────────────┐
        │  Azure Blob       │    │  MySQL Database         │
        │  Storage          │    │                         │
        │  - Features only  │    │  show_metadata          │
        │  - ~213 MB        │    │  show_similarities      │
        │                   │    │  (top 20 per show)      │
        └───────────────────┘    └────────┬────────────────┘
                                          │
                                          │
                                     ┌────▼───────────────────┐
                                     │  Azure Functions (API) │
                                     │                        │
                                     │  GET /shows/{id}/      │
                                     │      recommendations   │
                                     │                        │
                                     │  Queries MySQL for     │
                                     │  pre-computed results  │
                                     │  (<100ms response)     │
                                     └────────────────────────┘
```

**Key Design Decision:** Similarities are computed in-memory during database population and stored directly in MySQL. This avoids storing ~205GB of similarity matrices in blob storage while maintaining fast API response times.

## Project Structure

```
.
├── tvbingefriend_recommendation_service/
│   ├── ml/                     # Machine learning modules
│   │   ├── text_processor.py   # Text cleaning utilities
│   │   ├── feature_extractor.py # Feature engineering
│   │   └── similarity_computer.py # Similarity computation
│   ├── storage/                # Azure Blob Storage integration
│   │   └── blob_storage.py     # Blob client wrapper
│   ├── services/               # Business logic
│   │   ├── content_based_service.py # Recommendation service
│   │   └── data_loader_service.py   # API data fetching
│   ├── repos/                  # Database repositories
│   ├── models/                 # SQLAlchemy models
│   ├── blueprints/             # Azure Function blueprints
│   └── config.py               # Configuration management
├── scripts/                    # Production data pipeline
│   ├── fetch_and_prepare_data.py  # Step 1: Fetch shows
│   ├── compute_features.py        # Step 2: Feature extraction
│   ├── compute_similarities.py    # Step 3: Similarity computation
│   ├── populate_database.py       # Step 4: Database population
│   ├── run_pipeline.py            # Master pipeline script
│   ├── upload_to_blob.py          # Upload to Blob Storage
│   └── download_from_blob.py      # Download from Blob Storage
├── notebooks/                  # Development notebooks (not for production)
├── Dockerfile.pipeline         # Container image for data pipeline
└── function_app.py             # Azure Functions entry point
```

## Setup

### Prerequisites

- Python 3.12+
- Poetry
- Azure Storage Account
- MySQL Database
- Access to show/season/episode microservices

### Installation

```bash
# Install dependencies
poetry install

# Set up environment variables (see Configuration section)
cp local.settings.json.example local.settings.json
# Edit local.settings.json with your values
```

### Database Setup

```bash
# Run migrations
poetry run alembic upgrade head
```

## Configuration

Create `local.settings.json` for local development (Azure Functions API):

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "DATABASE_URL": "mysql+pymysql://user:password@localhost:3306/tvbingefriend_recommendations"
  }
}
```

**Note:** The API only needs database access. For the data pipeline environment variables, see the "Required for Data Pipeline" section below.

### Environment Variables

#### Required for Data Pipeline (Azure Container Instance)

- `AZURE_STORAGE_CONNECTION_STRING` - Azure Storage connection string
- `DATABASE_URL` - MySQL connection string
- `SHOW_SERVICE_URL` - Show service API endpoint
- `STORAGE_CONTAINER_NAME` - Blob container name (default: `recommendation-data`)
- `USE_BLOB_STORAGE` - Set to `true` for production

#### Required for Azure Functions

- `DATABASE_URL` - MySQL connection string

**Note:** Azure Functions only reads from the MySQL database. Blob storage variables are not required for the API.

## Local Development

### Running the Data Pipeline Locally

For testing and development, run the pipeline locally with a limited dataset:

```bash
# Run the complete pipeline with 100 shows (for testing)
poetry run python scripts/run_pipeline.py --max-shows 100
```

The pipeline automatically:
1. Fetches shows from the API
2. Computes features (genre, text, metadata)
3. Runs database migrations
4. Computes similarities in-memory (one show at a time)
5. Stores top N recommendations in the database

**Note:** The pipeline is optimized for memory efficiency - it computes similarities incrementally and stores only the top 20 recommendations per show in the database. Full similarity matrices are never saved to disk.

## Running Azure Functions Locally

```bash
# Start the function app
func start

# Or with Poetry
poetry run func start
```

### API Endpoints

- `GET /shows/{show_id}/recommendations?n=10&min_similarity=0.0`
- `GET /recommendations/stats`
- `GET /recommendations/health`

## Deployment

All infrastructure is provisioned and deployed automatically via Terraform and GitHub Actions. See the `.github/workflows/` directory for CI/CD configuration and the `terraform/` directory for infrastructure as code.

### Data Pipeline

The data pipeline runs weekly as an Azure Container Instance, triggered on a schedule by an Azure Logic App:
- **Schedule**: Weekly on Sunday at 9:00 AM UTC (5:00 AM Eastern Time)
- **Runs after**: Show/season/episode services complete their updates (2-4 AM ET)
- **Resources**: 2 CPU cores, 4 GB memory (configurable)
- **Notifications**: Optional email alerts on completion/failure (configured via Terraform variable)

The Docker image is built from `Dockerfile.pipeline` and pushed to Azure Container Registry automatically by GitHub Actions on every push to `main`.

### API

The Azure Functions app is deployed automatically via GitHub Actions:
- **Runtime**: Python 3.12
- **Plan**: Consumption or Premium
- **Data Source**: MySQL database (pre-computed recommendations)

### Configuration

All deployment configuration is managed through Terraform variables in `terraform/variables.tf`. See `.github/workflows/README.md` for required GitHub secrets and setup instructions.

## Blob Storage Structure

**Storage Optimized:** Only features are stored (~213 MB), not full similarity matrices (~205 GB)

```
recommendation-data/          # Container
└── processed/               # Blob prefix
    ├── shows_metadata.csv           # Show metadata
    ├── genre_features.npy           # Genre features
    ├── text_features.npz            # TF-IDF features (sparse)
    ├── platform_features.npy        # Platform features
    ├── type_features.npy            # Show type features
    ├── language_features.npy        # Language features
    ├── tfidf_vectorizer.pkl         # TF-IDF model
    └── genre_encoder.pkl            # Genre encoder

Total: ~213 MB (~$0.004/month in Azure Blob Storage)
```

**Note:** Similarity matrices are computed in-memory during pipeline execution and stored directly in the database. This saves ~205 GB of blob storage (from $3.69/month to $0.004/month for 80k shows).

## Feature Engineering

The system extracts multiple feature types:

1. **Genre Features** - Multi-hot encoding of genres
2. **Text Features** - TF-IDF on show summaries (500 features)
3. **Platform Features** - One-hot encoding of networks/streaming services
4. **Type Features** - Show type (Scripted, Reality, Animation)
5. **Language Features** - Primary language

### Similarity Computation

Similarities are computed using cosine similarity with weighted combination:

- **Genre Similarity** (40% weight) - Jaccard-like similarity on genres
- **Text Similarity** (50% weight) - Cosine similarity on TF-IDF vectors
- **Metadata Similarity** (10% weight) - Combined platform/type/language similarity

## Database Schema

### `show_metadata`
- Cached show information for fast API responses
- Fields: show_id, name, genres, summary, rating, type, language, network

### `show_similarities`
- Pre-computed similarity scores (top N per show)
- Fields: show_id, similar_show_id, similarity_score, genre_score, text_score, metadata_score
- Indexed on show_id for fast lookups

## Performance Considerations

### Cold Start Optimization

The Azure Functions app loads similarity matrices lazily to minimize cold start time. When using blob storage, matrices are downloaded once and cached in temp storage for the duration of the function instance.

### Database Caching

Pre-computed similarities are stored in MySQL for instant retrieval. The API typically responds in <100ms by querying the database directly.

### Pipeline Performance

For ~10,000 shows:
- Fetch & Prepare: ~10-15 minutes
- Feature Extraction: ~2-3 minutes
- Database Population (includes similarity computation): ~10-15 minutes
- **Total**: ~25-35 minutes

For ~80,000 shows:
- Fetch & Prepare: ~60-90 minutes
- Feature Extraction: ~5-10 minutes
- Database Population: ~60-90 minutes
- **Total**: ~2-3 hours

## Monitoring and Logging

### Pipeline Monitoring

The data pipeline automatically sends email notifications on completion or failure when `pipeline_notification_email` is configured in Terraform. Logs can be viewed in the Azure Portal under the Container Instance resource.

For local pipeline runs, logs are printed to stdout.

### Function Logs

Azure Functions logs are available in Azure Portal under Application Insights. For local development, use:
```bash
func logs
```

## Troubleshooting

### Issue: Pipeline fails at fetch step
- Verify `show_service_url` Terraform variable points to accessible API
- Check container logs in Azure Portal for error details
- For local testing, reduce `--max-shows` to test with smaller dataset

### Issue: Out of memory during pipeline execution
- Increase `pipeline_cpu_cores` or `pipeline_memory_in_gb` Terraform variables
- Check container logs to identify memory bottleneck

### Issue: Functions return empty recommendations
- Verify `show_similarities` table is populated (check database directly)
- Ensure pipeline has completed successfully (check email notification or Azure Portal)
- Verify database connection string is correct in Function App settings

### Issue: Slow recommendation API responses
- Ensure database indexes are created (Alembic migrations run automatically)
- Verify `show_similarities` table has data
- Consider upgrading to Premium App Service Plan for better performance

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Disclaimer

This project uses date from [TV Maze](https://www.tvmaze.com/) and the [TV Maze API](https://www.tvmaze.com/api), but it is not created by, managed by, endorsed by, or in any way affiliated with TV Maze.
