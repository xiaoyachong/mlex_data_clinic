import json
import logging
import os

import dash
import dash_bootstrap_components as dbc
import diskcache
from dash import dcc, html
from dash.long_callback import DiskcacheLongCallbackManager
from dotenv import load_dotenv
from file_manager.main import FileManager
from mlex_utils.dash_utils.mlex_components import MLExComponents

from src.components.header import header
from src.components.infrastructure import create_infra_state_affix
from src.components.main_display import main_display
from src.components.sidebar import sidebar
from src.components.training_stats import training_stats_plot
from src.utils.mlflow_utils import MlflowAlgorithmClient

load_dotenv(".env")

USER = os.getenv("USER", "admin")
READ_DIR = os.getenv("READ_DIR", "data")
SPLASH_URL = os.getenv("SPLASH_URL")
DEFAULT_TILED_URI = os.getenv("DEFAULT_TILED_URI")
DEFAULT_TILED_SUB_URI = os.getenv("DEFAULT_TILED_SUB_URI")
DATA_TILED_KEY = os.getenv("DATA_TILED_KEY")
if DATA_TILED_KEY == "":
    DATA_TILED_KEY = None
MODELFILE_PATH = os.getenv("MODELFILE_PATH", "./examples/assets/models.json")
MODE = os.getenv("MODE", "dev")
PREFECT_TAGS = json.loads(os.getenv("PREFECT_TAGS", '["data-clinic"]'))

# Set up logging
logger = logging.getLogger("dataclinic.app_layout")

# SETUP DASH APP
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "../assets/mlex-style.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
]
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    long_callback_manager=long_callback_manager,
)
app.title = "Data Clinic"
app._favicon = "mlex.ico"

# SETUP FILE MANAGER
dash_file_explorer = FileManager(
    READ_DIR,
    open_explorer=False,
    api_key=DATA_TILED_KEY,
    logger=logger,
)
dash_file_explorer.init_callbacks(app)
file_explorer = dash_file_explorer.file_explorer

# GET MODELS
latent_space_models = MlflowAlgorithmClient()
latent_space_models.load_from_mlflow(algorithm_type="latent_space_extraction")

dim_reduction_models = MlflowAlgorithmClient()
dim_reduction_models.load_from_mlflow(algorithm_type="dimension_reduction")

# SETUP MLEx COMPONENTS
mlex_components = MLExComponents("dbc")
job_manager = mlex_components.get_job_manager(
    model_list=latent_space_models.modelname_list,
    mode=MODE,
    aio_id="data-clinic-jobs",
    prefect_tags=PREFECT_TAGS,
)

# DEFINE LAYOUT
app.layout = html.Div(
    [
        header(
            "MLExchange | Data Clinic", "https://github.com/mlexchange/mlex_data_clinic"
        ),
        dbc.Container(
            [
                sidebar(file_explorer, job_manager),
                main_display(training_stats_plot()),
                html.Div(id="dummy-output"),
                dcc.Store(id="current-target-size", data=[0, 0]),
                create_infra_state_affix(),
            ],
            fluid=True,
        ),
    ]
)
