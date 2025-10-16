"""Shared test fixtures and configuration for pytest."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, List
import tempfile
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from tvbingefriend_recommendation_service.models.base import Base
from tvbingefriend_recommendation_service.models.show_metdata import ShowMetadata
from tvbingefriend_recommendation_service.models.show_similarity import ShowSimilarity


# ===== Database Fixtures =====

@pytest.fixture(scope="function")
def test_db_engine():
    """Create an in-memory SQLite database engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def test_db_session(test_db_engine):
    """Create a database session for testing."""
    SessionLocal = sessionmaker(bind=test_db_engine)
    session = SessionLocal()
    yield session
    session.close()


# ===== Sample Data Fixtures =====

@pytest.fixture
def sample_show_data() -> Dict:
    """Sample show data for testing."""
    return {
        'id': 1,
        'name': 'Breaking Bad',
        'genres': ['Drama', 'Crime', 'Thriller'],
        'summary': '<p>A high school chemistry teacher turned methamphetamine producer.</p>',
        'summary_clean': 'A high school chemistry teacher turned methamphetamine producer.',
        'type': 'Scripted',
        'language': 'English',
        'status': 'Ended',
        'platform': 'AMC',
        'rating_avg': 9.5,
        'network': {'name': 'AMC'},
        'webchannel': None,
        'rating': {'average': 9.5},
        'premiered': '2008-01-20',
        'ended': '2013-09-29'
    }


@pytest.fixture
def sample_shows_list() -> List[Dict]:
    """List of sample shows for testing."""
    return [
        {
            'id': 1,
            'name': 'Breaking Bad',
            'genres': ['Drama', 'Crime', 'Thriller'],
            'summary': '<p>A high school chemistry teacher.</p>',
            'summary_clean': 'A high school chemistry teacher.',
            'type': 'Scripted',
            'language': 'English',
            'status': 'Ended',
            'platform': 'AMC',
            'rating_avg': 9.5
        },
        {
            'id': 2,
            'name': 'Better Call Saul',
            'genres': ['Drama', 'Crime'],
            'summary': '<p>The trials and tribulations of criminal lawyer Jimmy McGill.</p>',
            'summary_clean': 'The trials and tribulations of criminal lawyer Jimmy McGill.',
            'type': 'Scripted',
            'language': 'English',
            'status': 'Ended',
            'platform': 'AMC',
            'rating_avg': 8.8
        },
        {
            'id': 3,
            'name': 'The Office',
            'genres': ['Comedy'],
            'summary': '<p>A mockumentary on a group of typical office workers.</p>',
            'summary_clean': 'A mockumentary on a group of typical office workers.',
            'type': 'Scripted',
            'language': 'English',
            'status': 'Ended',
            'platform': 'NBC',
            'rating_avg': 8.9
        }
    ]


@pytest.fixture
def sample_shows_df(sample_shows_list) -> pd.DataFrame:
    """Sample shows DataFrame for testing."""
    return pd.DataFrame(sample_shows_list)


@pytest.fixture
def sample_enriched_shows() -> List[Dict]:
    """Sample enriched shows with season data."""
    return [
        {
            'show_id': 1,
            'name': 'Breaking Bad',
            'summary': 'A high school chemistry teacher.',
            'genres': ['Drama', 'Crime'],
            'network': 'AMC',
            'webchannel': None,
            'rating': 9.5,
            'type': 'Scripted',
            'language': 'English',
            'status': 'Ended',
            'season_summaries': ['Season 1 summary', 'Season 2 summary'],
            'season_count': 5,
            'premiered': '2008-01-20',
            'ended': '2013-09-29'
        }
    ]


@pytest.fixture
def sample_genre_features() -> np.ndarray:
    """Sample genre feature matrix."""
    # 3 shows x 5 genres (one-hot encoded)
    return np.array([
        [1, 1, 1, 0, 0],  # Show 1: Drama, Crime, Thriller
        [1, 1, 0, 0, 0],  # Show 2: Drama, Crime
        [0, 0, 0, 1, 0],  # Show 3: Comedy
    ], dtype=float)


@pytest.fixture
def sample_text_features():
    """Sample text feature matrix (sparse)."""
    from scipy.sparse import csr_matrix
    # 3 shows x 10 text features
    data = np.array([0.5, 0.3, 0.7, 0.4, 0.6, 0.8])
    row = np.array([0, 0, 1, 1, 2, 2])
    col = np.array([0, 3, 1, 4, 2, 5])
    return csr_matrix((data, (row, col)), shape=(3, 10))


@pytest.fixture
def sample_platform_features() -> np.ndarray:
    """Sample platform feature matrix."""
    # 3 shows x 3 platforms (one-hot encoded with "other" category)
    return np.array([
        [1, 0, 0],  # Show 1: AMC
        [1, 0, 0],  # Show 2: AMC
        [0, 1, 0],  # Show 3: NBC
    ], dtype=float)


@pytest.fixture
def sample_type_features() -> np.ndarray:
    """Sample type feature matrix."""
    return np.array([
        [1, 0],  # Show 1: Scripted
        [1, 0],  # Show 2: Scripted
        [1, 0],  # Show 3: Scripted
    ], dtype=float)


@pytest.fixture
def sample_language_features() -> np.ndarray:
    """Sample language feature matrix."""
    return np.array([
        [1, 0],  # Show 1: English
        [1, 0],  # Show 2: English
        [1, 0],  # Show 3: English
    ], dtype=float)


@pytest.fixture
def sample_similarity_matrix() -> np.ndarray:
    """Sample similarity matrix."""
    return np.array([
        [1.0, 0.8, 0.2],
        [0.8, 1.0, 0.3],
        [0.2, 0.3, 1.0]
    ])


# ===== Mock Fixtures =====

@pytest.fixture
def mock_tfidf_vectorizer():
    """Mock TfidfVectorizer."""
    from scipy.sparse import csr_matrix
    mock = Mock()
    mock.fit_transform.return_value = csr_matrix(np.random.rand(3, 10))
    mock.get_feature_names_out.return_value = [f'word_{i}' for i in range(10)]
    return mock


@pytest.fixture
def mock_genre_encoder():
    """Mock MultiLabelBinarizer."""
    mock = Mock()
    mock.fit_transform.return_value = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    mock.classes_ = ['Drama', 'Comedy', 'Action']
    return mock


@pytest.fixture
def mock_blob_storage_client():
    """Mock BlobStorageClient."""
    mock = Mock()
    mock.upload_file.return_value = True
    mock.download_file.return_value = True
    mock.file_exists.return_value = True
    mock.list_blobs.return_value = ['file1.npy', 'file2.csv']
    mock.delete_file.return_value = True
    mock.upload_directory.return_value = 5
    mock.download_directory.return_value = 5
    return mock


@pytest.fixture
def mock_database_session():
    """Mock database session."""
    mock_session = Mock(spec=Session)
    mock_session.query.return_value = mock_session
    mock_session.filter.return_value = mock_session
    mock_session.first.return_value = None
    mock_session.all.return_value = []
    mock_session.count.return_value = 0
    mock_session.delete.return_value = None
    mock_session.commit.return_value = None
    mock_session.close.return_value = None
    mock_session.refresh.return_value = None
    return mock_session


@pytest.fixture
def mock_requests_session(requests_mock):
    """Mock requests session with requests_mock."""
    return requests_mock


# ===== Temporary Directory Fixtures =====

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_data_dir_with_files(temp_data_dir, sample_shows_df):
    """Create a temporary directory with sample data files."""
    # Save sample metadata
    metadata_path = temp_data_dir / 'shows_metadata.csv'
    sample_shows_df.to_csv(metadata_path, index=False)

    # Save sample features
    np.save(temp_data_dir / 'genre_features.npy', np.random.rand(3, 5))
    np.save(temp_data_dir / 'platform_features.npy', np.random.rand(3, 3))
    np.save(temp_data_dir / 'type_features.npy', np.random.rand(3, 2))
    np.save(temp_data_dir / 'language_features.npy', np.random.rand(3, 2))

    # Save sparse text features
    from scipy.sparse import save_npz, csr_matrix
    text_features = csr_matrix(np.random.rand(3, 10))
    save_npz(temp_data_dir / 'text_features.npz', text_features)

    # Save similarity matrices
    np.save(temp_data_dir / 'genre_similarity.npy', np.random.rand(3, 3))
    np.save(temp_data_dir / 'text_similarity.npy', np.random.rand(3, 3))
    np.save(temp_data_dir / 'metadata_similarity.npy', np.random.rand(3, 3))
    np.save(temp_data_dir / 'hybrid_similarity.npy', np.random.rand(3, 3))

    yield temp_data_dir


# ===== Configuration Fixtures =====

@pytest.fixture
def mock_config(monkeypatch):
    """Mock configuration values."""
    monkeypatch.setenv('DATABASE_URL', 'sqlite:///:memory:')
    monkeypatch.setenv('AZURE_STORAGE_CONNECTION_STRING', 'DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test==;EndpointSuffix=core.windows.net')
    monkeypatch.setenv('STORAGE_CONTAINER_NAME', 'test-container')
    monkeypatch.setenv('USE_BLOB_STORAGE', 'false')
    monkeypatch.setenv('SHOW_SERVICE_URL', 'http://localhost:7071/api')
    monkeypatch.setenv('SEASON_SERVICE_URL', 'http://localhost:7072/api')
    monkeypatch.setenv('EPISODE_SERVICE_URL', 'http://localhost:7073/api')


@pytest.fixture
def mock_local_settings(tmp_path, monkeypatch):
    """Mock local.settings.json file."""
    settings = {
        "Values": {
            "DATABASE_URL": "sqlite:///:memory:",
            "AZURE_STORAGE_CONNECTION_STRING": "test_connection_string",
            "STORAGE_CONTAINER_NAME": "test-container"
        }
    }

    settings_file = tmp_path / "local.settings.json"
    with open(settings_file, 'w') as f:
        json.dump(settings, f)

    # Mock the project root to point to tmp_path
    import tvbingefriend_recommendation_service.config as config_module
    monkeypatch.setattr(config_module, 'Path', lambda x: tmp_path if '__file__' in str(x) else Path(x))

    yield settings_file


# ===== Azure Functions Fixtures =====

@pytest.fixture
def mock_http_request():
    """Mock Azure Functions HttpRequest."""
    mock_req = Mock()
    mock_req.route_params = {}
    mock_req.params = {}
    mock_req.get_json.return_value = {}
    return mock_req


@pytest.fixture
def sample_show_metadata_records(test_db_session) -> List[ShowMetadata]:
    """Create sample ShowMetadata records in the test database."""
    records = [
        ShowMetadata(
            show_id=1,
            name='Breaking Bad',
            genres=['Drama', 'Crime'],
            summary='A chemistry teacher turns to cooking meth.',
            rating=9.5,
            type='Scripted',
            language='English',
            network='AMC'
        ),
        ShowMetadata(
            show_id=2,
            name='Better Call Saul',
            genres=['Drama', 'Crime'],
            summary='The story of Jimmy McGill.',
            rating=8.8,
            type='Scripted',
            language='English',
            network='AMC'
        ),
        ShowMetadata(
            show_id=3,
            name='The Office',
            genres=['Comedy'],
            summary='A mockumentary about office workers.',
            rating=8.9,
            type='Scripted',
            language='English',
            network='NBC'
        )
    ]

    for record in records:
        test_db_session.add(record)
    test_db_session.commit()

    return records


@pytest.fixture
def sample_similarity_records(test_db_session) -> List[ShowSimilarity]:
    """Create sample ShowSimilarity records in the test database."""
    records = [
        ShowSimilarity(
            show_id=1,
            similar_show_id=2,
            similarity_score=0.85,
            genre_score=0.9,
            text_score=0.8,
            metadata_score=0.85
        ),
        ShowSimilarity(
            show_id=1,
            similar_show_id=3,
            similarity_score=0.25,
            genre_score=0.1,
            text_score=0.3,
            metadata_score=0.4
        ),
        ShowSimilarity(
            show_id=2,
            similar_show_id=1,
            similarity_score=0.85,
            genre_score=0.9,
            text_score=0.8,
            metadata_score=0.85
        )
    ]

    for record in records:
        test_db_session.add(record)
    test_db_session.commit()

    return records


# ===== Azure Functions Fixtures =====

@pytest.fixture
def mock_azure_func_request():
    """Mock Azure Functions HttpRequest with common methods."""
    mock_req = Mock()
    mock_req.route_params = {}
    mock_req.params = {}
    mock_req.get_json.return_value = {}
    mock_req.get_body.return_value = b'{}'
    mock_req.method = 'GET'
    mock_req.url = 'http://localhost/api/test'
    return mock_req


# ===== Blob Storage Mock Fixtures =====

@pytest.fixture
def mock_storage_service():
    """Mock StorageService from tvbingefriend-azure-storage-service."""
    mock = Mock()
    mock_container_client = Mock()
    mock_blob_client = Mock()

    # Setup blob client methods
    mock_blob_client.exists.return_value = True
    mock_blob_client.upload_blob.return_value = None
    mock_blob_client.download_blob.return_value.readall.return_value = b'test data'
    mock_blob_client.delete_blob.return_value = None

    # Setup container client methods
    mock_container_client.get_blob_client.return_value = mock_blob_client
    mock_container_client.list_blobs.return_value = []

    # Setup storage service
    mock.get_blob_service_client.return_value = mock_container_client

    return mock


@pytest.fixture
def mock_blob_properties():
    """Mock blob properties for Azure blob storage."""
    mock_blob = Mock()
    mock_blob.name = 'test_blob.npy'
    mock_blob.size = 1024
    return mock_blob


# ===== Repository Fixtures =====

@pytest.fixture
def metadata_repository(test_db_session):
    """Create MetadataRepository with test database session."""
    from tvbingefriend_recommendation_service.repos import MetadataRepository
    return MetadataRepository(test_db_session)


@pytest.fixture
def similarity_repository(test_db_session):
    """Create SimilarityRepository with test database session."""
    from tvbingefriend_recommendation_service.repos import SimilarityRepository
    return SimilarityRepository(test_db_session)


# ===== Script Fixtures =====

@pytest.fixture
def mock_sys_argv(monkeypatch):
    """Mock sys.argv for script testing."""
    def _mock_argv(args):
        monkeypatch.setattr('sys.argv', args)
    return _mock_argv


@pytest.fixture
def mock_argparse():
    """Mock argparse for script testing."""
    mock_args = Mock()
    mock_args.output_dir = 'data/processed'
    mock_args.show_service_url = None
    mock_args.batch_size = 100
    mock_args.max_shows = None
    mock_args.input_dir = 'data/processed'
    return mock_args
