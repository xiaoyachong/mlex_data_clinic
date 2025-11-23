import json
import logging
import os
import traceback
import uuid
from datetime import datetime

import mlflow
import pytz
from dash import MATCH, Input, Output, State, callback, html, no_update
from dash.exceptions import PreventUpdate
from file_manager.data_project import DataProject
from mlex_utils.prefect_utils.core import (
    get_children_flow_run_ids,
    get_flow_run_name,
    get_flow_run_parameters,
    get_flow_run_state,
    schedule_prefect_flow,
)

from src.app_layout import (
    USER,
    dim_reduction_models,
    latent_space_models,
    mlex_components,
)
from src.utils.data_utils import tiled_results
from src.utils.job_utils import (
    parse_inference_job_params,
    parse_model_params,
    parse_train_job_params,
)
from src.utils.plot_utils import generate_notification

MODE = os.getenv("MODE", "")
TIMEZONE = os.getenv("TIMEZONE", "US/Pacific")
FLOW_NAME = os.getenv("FLOW_NAME", "")
PREFECT_TAGS = json.loads(os.getenv("PREFECT_TAGS", '["data-clinic"]'))
WRITE_DIR = os.getenv("WRITE_DIR", "")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

logger = logging.getLogger(__name__)


@callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "notifications-container",
            "aio_id": MATCH,
        },
        "children",
        allow_duplicate=True,
    ),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "train-button",
            "aio_id": MATCH,
        },
        "n_clicks",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-parameters",
            "aio_id": MATCH,
        },
        "children",
    ),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-list",
            "aio_id": MATCH,
        },
        "value",
    ),
    State("log-transform", "value"),
    State("min-max-percentile", "value"),
    State("mask-dropdown", "value"),
    State(
        {"component": "DbcJobManagerAIO", "subcomponent": "job-name", "aio_id": MATCH},
        "value",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "project-name-id",
            "aio_id": "data-clinic-jobs",
        },
        "data",
    ),
    prevent_initial_call=True,
)
def run_train(
    n_clicks,
    model_parameter_container,
    data_project_dict,
    model_name,
    log,
    percentiles,
    mask,
    job_name,
    project_name,
):
    """
    This callback submits a job request to the compute service
    Args:
        n_clicks:                   Number of clicks
        model_parameter_container:  App parameters
        data_project_dict:          Data project dictionary
        model_name:                 Selected model name
        log:                        Log transform
        percentiles:                Min-max percentiles
        mask:                       Mask selection
        job_name:                   Job name
        project_name:               Project name
    Returns:
        open the alert indicating that the job was submitted
    """
    if n_clicks is not None and n_clicks > 0:
        model_parameters, parameter_errors = parse_model_params(
            model_parameter_container, log, percentiles, mask
        )
        # Check if the model parameters are valid
        if parameter_errors:
            notification = generate_notification(
                "Model Parameters",
                "red",
                "fluent-mdl2:machine-learning",
                "Model parameters are not valid!",
            )
            return notification

        data_project = DataProject.from_dict(data_project_dict)

        latent_space_params = latent_space_models[model_name]
        dim_reduction_params = dim_reduction_models[
            "umap_v1.0.0"  # since there is no dim_reduction model seletion in UI, we use umap as default
        ]
        train_params = parse_train_job_params(
            data_project,
            model_parameters,
            USER,
            project_name,
            latent_space_params,
            dim_reduction_params,
        )

        if MODE == "dev":
            job_uid = str(uuid.uuid4())
            job_message = (
                f"Dev Mode: Job has been succesfully submitted with uid: {job_uid}"
            )
            notification_color = "primary"
        else:
            try:
                # Prepare tiled
                tiled_results.prepare_project_container(USER, project_name)
                # Schedule job
                current_time = datetime.now(pytz.timezone(TIMEZONE)).strftime(
                    "%Y/%m/%d %H:%M:%S"
                )
                job_uid = schedule_prefect_flow(
                    FLOW_NAME,
                    parameters=train_params,
                    flow_run_name=f"{job_name} {current_time}",
                    tags=PREFECT_TAGS + ["train", project_name],
                )
                job_message = f"Job has been succesfully submitted with uid: {job_uid}"
                notification_color = "indigo"
            except Exception as e:
                # Print the traceback to the console
                logger.error(f"Error submitting job: {e} \n{traceback.format_exc()}")
                job_uid = None
                job_message = f"Job presented error: {e}"
                notification_color = "danger"

        notification = generate_notification(
            "Job Submission", notification_color, "formkit:submit", job_message
        )

        return notification
    raise PreventUpdate


@callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-parameters",
            "aio_id": MATCH,
        },
        "children",
        allow_duplicate=True,
    ),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "train-dropdown",
            "aio_id": MATCH,
        },
        "value",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-parameters",
            "aio_id": MATCH,
        },
        "children",
    ),
    prevent_initial_call=True,
)
def update_model_parameters(job_id, current_params):
    try:
        job_parameters = get_flow_run_parameters(job_id)
    except Exception as e:
        logger.error(f"Error submitting job: {e} \n{traceback.format_exc()}")
        raise PreventUpdate
    train_parameters = job_parameters["params_list"][0]["params"]
    model_parameters = train_parameters["model_parameters"]
    mlex_components.__init__("dbc")
    return mlex_components.update_parameters_values(current_params, model_parameters)


@callback(
    Output("show-reconstructions", "disabled"),
    Output("show-reconstructions", "value"),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "train-dropdown",
            "aio_id": "data-clinic-jobs",
        },
        "value",
    ),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "check-job",
            "aio_id": "data-clinic-jobs",
        },
        "n_intervals",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "project-name-id",
            "aio_id": "data-clinic-jobs",
        },
        "data",
    ),
    prevent_initial_call=True,
)
def allow_show_reconstructions(job_id, n_intervals, project_name):
    """
    Determine whether to show reconstructions for the selected job. This callback checks whether a
    given job has completed and whether its reconstruction data is available.
    Args:
        job_id:             Selected job
        n_intervals:        Number of intervals
        project_name:       Data project name
    Returns:
        rec_img:            Reconstructed image
    """
    if job_id is not None:
        children_job_ids = get_children_flow_run_ids(job_id)

        if (
            len(children_job_ids) != 3
            or get_flow_run_state(children_job_ids[1]) != "COMPLETED"
        ):
            return True, False

        recons_child_job_id = children_job_ids[1]
        expected_recons_uri = (
            f"{USER}/{project_name}/{recons_child_job_id}/reconstructions"
        )
        latent_vector_child_job_id = children_job_ids[2]
        expected_latent_vector_uri = (
            f"{USER}/{project_name}/{latent_vector_child_job_id}"
        )
        try:
            tiled_results.get_data_by_trimmed_uri(expected_recons_uri)
            tiled_results.get_data_by_trimmed_uri(expected_latent_vector_uri).read()
            return False, no_update
        except Exception as e:
            logger.error(f"Error submitting job: {e} \n{traceback.format_exc()}")
            return True, False
    else:
        return True, False


@callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "show-training-stats",
            "aio_id": "data-clinic-jobs",
        },
        "disabled",
    ),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "train-dropdown",
            "aio_id": "data-clinic-jobs",
        },
        "value",
    ),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "check-job",
            "aio_id": "data-clinic-jobs",
        },
        "n_intervals",
    ),
    prevent_initial_call=True,
)
def allow_show_stats(job_id, check_job_n_intervals):
    if job_id is not None:
        children_job_ids = get_children_flow_run_ids(job_id)

        if len(children_job_ids) == 0 or get_flow_run_state(
            children_job_ids[0]
        ) not in ["COMPLETED", "RUNNING"]:
            return True

        child_job_id = children_job_ids[0]  # training job (Prefect flow run ID)

        # Check if the report file exists in MLflow
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()
            
            # Search for model by Prefect job ID
            model_versions = client.search_model_versions(
                filter_string=f"name='{child_job_id}'"
            )
            if not model_versions or len(model_versions) == 0:
                return True
            
            # Get the run_id from the model version
            mlflow_run_id = model_versions[0].run_id
            artifacts = client.list_artifacts(mlflow_run_id, "dvc_metrics")
            if any(artifact.path == "dvc_metrics/report.html" for artifact in artifacts):
                return False
        except Exception as e:
            logger.error(f"Error checking MLflow artifacts: {e}")
            return True

    return True


@callback(
    Output("stats-card-body", "children"),
    Output("show-plot", "is_open"),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "show-training-stats",
            "aio_id": "data-clinic-jobs",
        },
        "n_clicks",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "train-dropdown",
            "aio_id": "data-clinic-jobs",
        },
        "value",
    ),
    prevent_initial_call=True,
)
def show_training_stats(show_stats_n_clicks, job_id):
    if show_stats_n_clicks > 0:

        children_job_ids = get_children_flow_run_ids(job_id)
        child_job_id = children_job_ids[0]  # Prefect flow run ID
        
        # Download report from MLflow
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()
            
            # Search for model by Prefect job ID
            model_versions = client.search_model_versions(
                filter_string=f"name='{child_job_id}'"
            )
            if not model_versions or len(model_versions) == 0:
                logger.error(f"No model found for Prefect flow: {child_job_id}")
                return [], False
            
            # Get the run_id from the model version
            mlflow_run_id = model_versions[0].run_id
            logger.info(f"Found MLflow run ID: {mlflow_run_id} for Prefect flow: {child_job_id}")
            
            report_path = client.download_artifacts(mlflow_run_id, "dvc_metrics/report.html")
            
            with open(report_path, "r") as f:
                report_html = f.read()

            return (
                html.Iframe(srcDoc=report_html, style={"width": "100%", "height": "600px"}),
                True,
            )
        except Exception as e:
            logger.error(f"Error loading report from MLflow: {e}")
            return [], False
    else:
        return [], False


@callback(
    Output(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "notifications-container",
            "aio_id": MATCH,
        },
        "children",
        allow_duplicate=True,
    ),
    Input(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "inference-button",
            "aio_id": MATCH,
        },
        "n_clicks",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "train-dropdown",
            "aio_id": "data-clinic-jobs",
        },
        "value",
    ),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-parameters",
            "aio_id": MATCH,
        },
        "children",
    ),
    State({"base_id": "file-manager", "name": "data-project-dict"}, "data"),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "model-list",
            "aio_id": MATCH,
        },
        "value",
    ),
    State("log-transform", "value"),
    State("min-max-percentile", "value"),
    State("mask-dropdown", "value"),
    State(
        {
            "component": "DbcJobManagerAIO",
            "subcomponent": "project-name-id",
            "aio_id": "data-clinic-jobs",
        },
        "data",
    ),
    prevent_initial_call=True,
)
def run_inference(
    n_clicks,
    train_job_id,
    model_parameter_container,
    data_project_dict,
    model_name,
    log,
    percentiles,
    mask,
    project_name,
):
    """
    This callback submits a job request to the compute service
    Args:
        n_clicks:                   Number of clicks
        train_job_id:               Selected training job
        model_parameter_container:  App parameters
        data_project_dict:          Data project dictionary
        model_name:                 Selected model name
        log:                        Log transform
        percentiles:                Min-max percentiles
        mask:                       Mask selection
        project_name:               Project name
    Returns:
        open the alert indicating that the job was submitted
    """
    if n_clicks is not None and n_clicks > 0:
        model_parameters, parameter_errors = parse_model_params(
            model_parameter_container, log, percentiles, mask
        )
        # Check if the model parameters are valid
        if parameter_errors:
            notification = generate_notification(
                "Model Parameters",
                "red",
                "fluent-mdl2:machine-learning",
                "Model parameters are not valid!",
            )
            return notification

        data_project = DataProject.from_dict(data_project_dict)
        latent_space_params = latent_space_models[model_name]
        dim_reduction_params = dim_reduction_models[
            "umap_v1.0.0"  # since there is no dim_reduction model seletion in UI, we use umap as default
        ]
        inference_params = parse_inference_job_params(
            data_project,
            model_parameters,
            USER,
            project_name,
            latent_space_params,
            dim_reduction_params,
        )

        if MODE == "dev":
            job_uid = str(uuid.uuid4())
            job_message = (
                f"Dev Mode: Job has been succesfully submitted with uid: {job_uid}"
            )
            notification_color = "primary"
        else:
            if (
                train_job_id is not None
                and get_flow_run_state(train_job_id) == "COMPLETED"
            ):
                job_name = get_flow_run_name(train_job_id)
                children_flows = get_children_flow_run_ids(train_job_id)
                train_job_id = children_flows[0]
                inference_params["params_list"][0]["params"]["io_parameters"][
                    "uid_retrieve"
                ] = train_job_id

                try:
                    # Prepare tiled
                    tiled_results.prepare_project_container(USER, project_name)
                    # Schedule job
                    current_time = datetime.now(pytz.timezone(TIMEZONE)).strftime(
                        "%Y/%m/%d %H:%M:%S"
                    )
                    job_uid = schedule_prefect_flow(
                        FLOW_NAME,
                        parameters=inference_params,
                        flow_run_name=f"{job_name} {current_time}",
                        tags=PREFECT_TAGS + ["inference", project_name],
                    )
                    job_message = (
                        f"Job has been succesfully submitted with uid: {job_uid}"
                    )
                    notification_color = "indigo"
                except Exception as e:
                    # Print the traceback to the console
                    logger.error(
                        f"Error submitting job: {e} \n{traceback.format_exc()}"
                    )
                    job_message = f"Job presented error: {e}"
                    notification_color = "danger"
            else:
                job_message = "Train job has not completed"
                notification_color = "warning"

            notification = generate_notification(
                "Job Submission", notification_color, "formkit:submit", job_message
            )

        return notification
    raise PreventUpdate
