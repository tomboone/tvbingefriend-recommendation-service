# TV BingeFriend Recommendation Service

Content-based TV show recommendation system using hybrid similarity scoring.

## Architecture Overview

### Components

1. **Azure Functions** - API layer for serving recommendations
2. **Azure Container Instance** - Data pipeline for processing and computing similarities
3. **Azure Blob Storage** - Storage for feature matrices and similarity data
4. **MySQL Database** - Storage for show metadata and pre-computed recommendations

### Data Flow

```
┌─────────────────────────────────────┐
│  Azure Container Instance           │
│  (Scheduled - Weekly/Monthly)       │
│                                     │
│  1. Fetch shows from API            │
│  2. Compute features                │
│  3. Compute similarities            │
│  4. Upload to Blob Storage          │
│  5. Populate MySQL Database         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Azure Blob Storage                 │
│  - Feature matrices (NPY/NPZ)       │
│  - Similarity matrices (NPY)        │
│  - ML models (PKL)                  │
│  - Show metadata (CSV)              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Azure Functions (API)              │
│                                     │
│  GET /shows/{id}/recommendations    │
│  - Loads from MySQL (fast)          │
│  - Falls back to Blob if needed     │
└─────────────────────────────────────┘
```

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

Create `local.settings.json` for local development:

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",

    "DATABASE_URL": "mysql+pymysql://user:password@localhost:3306/tvbingefriend_recommendations",

    "SHOW_SERVICE_URL": "http://localhost:7071/api",
    "SEASON_SERVICE_URL": "http://localhost:7072/api",
    "EPISODE_SERVICE_URL": "http://localhost:7073/api",

    "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net",
    "STORAGE_CONTAINER_NAME": "recommendation-data",
    "USE_BLOB_STORAGE": "false"
  }
}
```

### Environment Variables

#### Required for Data Pipeline (Azure Container Instance)

- `AZURE_STORAGE_CONNECTION_STRING` - Azure Storage connection string
- `DATABASE_URL` - MySQL connection string
- `SHOW_SERVICE_URL` - Show service API endpoint
- `STORAGE_CONTAINER_NAME` - Blob container name (default: `recommendation-data`)
- `USE_BLOB_STORAGE` - Set to `true` for production

#### Required for Azure Functions

- `DATABASE_URL` - MySQL connection string
- `AZURE_STORAGE_CONNECTION_STRING` - Azure Storage connection string (if using blob storage)
- `USE_BLOB_STORAGE` - Set to `true` to load from blob storage

## Running the Data Pipeline

### Local Development (with local files)

```bash
# Step 1: Fetch and prepare data
poetry run python scripts/fetch_and_prepare_data.py --max-shows 100

# Step 2: Compute features
poetry run python scripts/compute_features.py

# Step 3: Populate database (computes similarities in-memory, stores in DB)
poetry run python scripts/populate_database.py

# OR run the entire pipeline at once
poetry run python scripts/run_pipeline.py --max-shows 100
```

**Note:** The pipeline automatically optimizes storage by computing similarities in-memory and storing only top N recommendations in the database. Full similarity matrices are never saved to disk/blob storage.

### Production (with Azure Blob Storage)

```bash
# Run full pipeline and upload to blob storage
poetry run python scripts/run_pipeline.py

# Upload processed data to blob storage
poetry run python scripts/upload_to_blob.py

# Download processed data from blob storage (for testing)
poetry run python scripts/download_from_blob.py
```

### Pipeline Options

The pipeline supports various configuration options:

```bash
# Fetch only specific number of shows (for testing)
python scripts/run_pipeline.py --max-shows 1000

# Run specific steps only
python scripts/run_pipeline.py --steps fetch,features

# Customize similarity weights
python scripts/run_pipeline.py \
  --genre-weight 0.5 \
  --text-weight 0.4 \
  --metadata-weight 0.1

# Adjust top N recommendations per show
python scripts/run_pipeline.py --top-n 30 --min-similarity 0.15
```

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

## Docker

### Build Pipeline Image

```bash
# Build the data pipeline container
docker build -f Dockerfile.pipeline -t recommendation-pipeline:latest .

# Run locally (with environment variables)
docker run --env-file .env recommendation-pipeline:latest
```

## Deployment

### Azure Container Instance (Data Pipeline)

The data pipeline should be deployed as an Azure Container Instance, configured to run on a schedule (weekly or monthly) via your Terraform configuration.

**Key Terraform Resources Needed:**
- `azurerm_container_group` - The container instance
- Environment variables from your Terraform outputs (storage connection, database URL, etc.)
- Restart policy: `OnFailure`

**Container Configuration:**
- Image: Built from `Dockerfile.pipeline`
- CPU: 2-4 cores recommended
- Memory: 4-8 GB recommended (depends on dataset size)
- Command: `python scripts/run_pipeline.py` (or customize with specific steps)

### Azure Functions (API)

Deploy as a standard Azure Functions app using your existing Terraform and GitHub Actions setup.

**Key Configuration:**
- Runtime: Python 3.12
- Plan: Consumption or Premium (for better cold start performance)
- App Settings: Map environment variables from Terraform outputs

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

### Pipeline Logs

When running in Azure Container Instance:
```bash
# View logs
az container logs --resource-group <rg> --name <container-name>

# Stream live logs
az container attach --resource-group <rg> --name <container-name>
```

### Function Logs

View logs in Azure Portal under Application Insights or use:
```bash
func logs
```

## Troubleshooting

### Issue: Pipeline fails at fetch step
- Verify `SHOW_SERVICE_URL` is accessible
- Check API authentication if required
- Reduce `--max-shows` for testing

### Issue: Out of memory during similarity computation
- Increase container memory allocation
- Process shows in batches (modify scripts)
- Reduce `--max-text-features`

### Issue: Functions can't load from blob storage
- Verify `AZURE_STORAGE_CONNECTION_STRING` is set
- Check blob container exists and has data
- Verify `USE_BLOB_STORAGE=true` is set

### Issue: Slow recommendation API responses
- Ensure database indexes are created
- Check if `show_similarities` table is populated
- Consider increasing Function App tier

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Disclaimer

This project uses date from [TV Maze](https://www.tvmaze.com/) and the [TV Maze API](https://www.tvmaze.com/api), but it is not created by, managed by, endorsed by, or in any way affiliated with TV Maze.
