"""Example workflow pipeline script for Edge CV pipeline.
                                                                                 . -ModelStep
                                                                                .
    Process-> Train -> Evaluate -> Condition .
                                                  |                              .
                                                  |                                . -(stop)
                                                  |
                                                   -> CreateModel

Implements a get_pipeline(**kwargs) method.
"""
import os
import time 
import logging

import boto3
import sagemaker
import sagemaker.session

from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.pytorch.processing import PyTorchProcessor

from sagemaker.inputs import TrainingInput, CreateModelInput, TransformInput
from sagemaker.model import Model
from sagemaker.transformer import Transformer

from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
    FileSource
)

from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)

from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep,
    TransformStep,
    CacheConfig
)
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)

from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.model_monitor import DatasetFormat, model_monitoring
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    """Gets the sagemaker client.
        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn.lower())
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket="sagemaker-rfml-edge",
    pipeline_model_name="clouddrf-pipeline",
    model_package_group_name="clouddrf-model-group",
    pipeline_name="clouddrf-Pipeline",
    sagemaker_project_name="cloudd-rf",
    base_job_prefix="cloudd-rf"
):
    """Gets a SageMaker ML Pipeline instance working with on rf data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    
    ################ SAGEMAKER SESSION CONFIG #####################
    sagemaker_session = get_session(region, default_bucket)
    
    if default_bucket is None:
        default_bucket = sagemaker_session.default_bucket()

    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)
    
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")
    
    code_dir = "code"
    
    ################ SAGEMAKER PIPELINE PARAMETERS #####################    
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")
    
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)    
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.c5.4xlarge")

    fusion_instance_type = ParameterString(name="FusionInstanceType", default_value="ml.g5.xlarge")
    
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.g5.xlarge")    
    
    inference_instance_type = ParameterString(name="InferenceInstanceType", default_value="ml.m5.large")
    
    chunk_size = ParameterInteger(name="ChunkSize", default_value=1000)
    sample_size = ParameterInteger(name="SampleSize", default_value=100000)
    num_sensors = ParameterInteger(name="NumSensors", default_value=4)
    batch_size = ParameterInteger(name="BatchSize", default_value=1)
    
    ################ PROCESSING STEP #####################
    timestamp = str(time.time()).split('.')[0]
    
    output_prefix = f'{base_job_prefix}/preprocess/outputs/{timestamp}'
    process_output_s3_uri = f's3://{default_bucket}/{output_prefix}'
    process_output_local = '/opt/ml/processing/output'
    
    code_location = f's3://{default_bucket}/{base_job_prefix}/preprocess/code'
    
    train_split = 0.7
    val_split = 0.1
    test_split = 0.2
    max_trials = 1000
    max_sigs = 1
    allow_collisions = "False"
    image_width = 1000
    image_height = 500
    fft_size = 256
    overlap = 255
    rand_seed = 1337

    arguments = [
        "--chunk-size", chunk_size.to_string(), 
        "--train-split", str(train_split), 
        "--val-split", str(val_split), 
        "--test-split", str(test_split),
        "--sample-size", sample_size.to_string(),
        "--max-sigs", str(max_sigs),
        "--max-trials", str(max_trials),
        "--allow-collisions", allow_collisions,
        "--image-width", str(image_width),
        "--image-height", str(image_height),
        "--fft-size", str(fft_size),
        "--overlap", str(overlap),
        "--rand-seed", str(rand_seed)
    ]
    
    outputs=[ 
        ProcessingOutput(output_name="train", source=f"{process_output_local}/train", destination = process_output_s3_uri +'/train'),
        ProcessingOutput(output_name="validation", source=f"{process_output_local}/validation", destination = process_output_s3_uri +'/validation'),
        ProcessingOutput(output_name="test", source=f"{process_output_local}/test", destination = process_output_s3_uri +'/test')
    ]
    
    pytorch_processor = PyTorchProcessor(
        framework_version='1.13.1',
        py_version="py39",
        role=role,
        volume_size_in_gb = 30,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name = f"{base_job_prefix}-preprocess",
        sagemaker_session=pipeline_session
    )
    
    step_process = ProcessingStep(
        name="PreprocessData",
        cache_config=cache_config,
        step_args = pytorch_processor.run(
            outputs=outputs,
            arguments=arguments,
            source_dir=code_dir,
            code="preprocessing.py"
        )
    )
    
    ################ TRAINING STEP #####################
    output_prefix = f'{base_job_prefix}/training'

    timestamp = str(time.time()).split('.')[0]
    code_location = f"s3://{default_bucket}/{output_prefix}/{timestamp}/code"
    
    # Be sure to update the chunk-size and obs-int hyperparameters so the IQ data gets parsed correctly
    hyperparameters = {
                        "lr": 0.001,
                        "batch_size": batch_size,
                        "epochs": 10,
                        "chunk-size": chunk_size
                    }

    metric_definitions = [{'Name': 'loss',      'Regex': "'loss': ([0-9\\.]+)"},
                          {'Name': 'recall',       'Regex': "'recall': ([0-9\\.]+)"},
                          {'Name': 'map50',  'Regex': "'map50': ([0-9\\.]+)"},
                          {'Name': 'map',   'Regex': "'map': ([0-9\\.]+)"}]


    distributions = {'parameter_server': {'enabled': False}}
    DISTRIBUTION_MODE = 'FullyReplicated'
    train_script = 'train.py'

    # Set the training script related parameters
    container_log_level = logging.INFO

    # Location where the trained model will be stored locally in the container before being uploaded to S3
    model_local_dir = '/opt/ml/model'
    model_path = f"s3://{default_bucket}/{output_prefix}"

    # Access the location where the preceding processing step saved train and validation datasets
    # Pipeline step properties can give access to outputs which can be used in succeeding steps
    team_data_in = TrainingInput(
                        s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                        distribution=DISTRIBUTION_MODE,
                        s3_data_type='S3Prefix',
                        input_mode='FastFile'
                    )
    inputs = {'team_data_dir': team_data_in}
    
    
    image_uri = image_uris.retrieve(framework='pytorch',region='us-east-1',version='1.13.1',py_version='py39',image_scope='training', instance_type=training_instance_type)
    
    # Create the estimator
    estimator = PyTorch(
                    entry_point=train_script,
                    source_dir=code_dir,
                    output_path=model_path,
                    distribution=distributions,
                    instance_type=training_instance_type,
                    instance_count=training_instance_count,
                    hyperparameters=hyperparameters,
                    metric_definitions=metric_definitions,
                    role=role,
                    enable_sagemaker_metrics=False,
                    disable_profiler=True,
                    debugger_hook_config=False,
                    base_job_name=f'{base_job_prefix}-training',
                    image_uri=image_uri,
                    container_log_level=container_log_level,
                    input_mode="FastFile",
                    script_mode=True,
                    disable_output_compression=True
               )
    
    # Set pipeline training step
    step_train = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        cache_config=cache_config,
        inputs=inputs,
        depends_on=["PreprocessData"]
    )
    
    ############ FEATURE CONTRIBUTION STEP ########################
    timestamp = str(time.time()).split('.')[0]
    output_prefix = f'{base_job_prefix}/analysis/feature_contribution/outputs/{timestamp}'
    output_s3_uri = f's3://{default_bucket}/{output_prefix}'
    code_location = f's3://{default_bucket}/{base_job_prefix}/analysis/feature_contribution/code'

    env_vars = {
        "SM_CHANNEL_TEST": "/opt/ml/processing/input/data/test",
        "SM_MODEL_DIR": "/opt/ml/processing/model",
        "SM_OUTPUT_DIR": "/opt/ml/processing/output"
    }
    
    # Processing Script Arguments
    arguments = [
        "--samples-per-batch", chunk_size.to_string(), 
        "--batch-size", batch_size.to_string(),
        "--num-sensors", num_sensors.to_string()
    ]

    code = 'feature_contribution.py'

    script_baseline_fusion = PyTorchProcessor(
        framework_version='1.13.1',
        py_version="py39",
        role=role,
        env=env_vars,
        instance_count=processing_instance_count,
        instance_type=fusion_instance_type,
        base_job_name = f"{base_job_prefix}-feature-contribution",
        code_location=code_location,
        sagemaker_session=pipeline_session
    )

    step_args = script_baseline_fusion.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination=env_vars["SM_MODEL_DIR"],
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination=env_vars["SM_CHANNEL_TEST"],
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="output", source=env_vars["SM_OUTPUT_DIR"], destination = output_s3_uri),
        ],
        arguments=arguments,
        source_dir=code_dir,
        code=code
    )
    
    step_feature_contribution = ProcessingStep(
        name="FeatureContribution",
        cache_config=cache_config,
        step_args = step_args,
        depends_on=["TrainModel"]
    )
    
    ############ FEATURE EXTRACTION STEP ########################
    timestamp = str(time.time()).split('.')[0]
    output_prefix = f'{base_job_prefix}/analysis/feature_extraction/outputs/{timestamp}'
    output_s3_uri = f's3://{default_bucket}/{output_prefix}'
    code_location = f's3://{default_bucket}/{base_job_prefix}/analysis/feature_extraction/code'

    env_vars = {
        "SM_CHANNEL_TEST": "/opt/ml/processing/input/data/test",
        "SM_MODEL_DIR": "/opt/ml/processing/model",
        "SM_OUTPUT_DIR": "/opt/ml/processing/output"
    }
    
    # Processing Script Arguments
    arguments = [
        "--samples-per-batch", chunk_size.to_string(), 
        "--batch-size", batch_size.to_string(),
        "--num-sensors", num_sensors.to_string()
    ]

    code = 'feature_extraction.py'

    script_baseline_fusion = PyTorchProcessor(
        framework_version='1.13.1',
        py_version="py39",
        role=role,
        env=env_vars,
        instance_count=processing_instance_count,
        instance_type=fusion_instance_type,
        base_job_name = f"{base_job_prefix}-feature-extraction",
        code_location=code_location,
        sagemaker_session=pipeline_session
    )

    step_args = script_baseline_fusion.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination=env_vars["SM_MODEL_DIR"],
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination=env_vars["SM_CHANNEL_TEST"],
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="output", source=env_vars["SM_OUTPUT_DIR"], destination = output_s3_uri),
        ],
        arguments=arguments,
        source_dir=code_dir,
        code=code
    )
    
    step_feature_extraction = ProcessingStep(
        name="FeatureExtraction",
        cache_config=cache_config,
        step_args = step_args,
        depends_on=["TrainModel"]
    )
    
    
    ############ MODEL CREATE STEP ########################
    #.to_string()
#     model = Model(
#         image_uri=image_uri,
#         model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
#         sagemaker_session=pipeline_session,
#         role=role,
#     )

#     step_args = model.create(
#         instance_type="ml.m5.large"
#     )

#     step_create_model = ModelStep(
#         name="TeamModel",
#         step_args=step_args,
#         depends_on=["TrainModel"]
#     )
    
#     step_args = model.register(
#         content_types=["application/json"],
#         response_types=["application/json"],
#         inference_instances=["ml.g5.xlarge", "ml.m5.large"],
#         transform_instances=["ml.g5.xlarge"],
#         model_package_group_name=model_package_group_name,
#         approval_status=model_approval_status
#     )
#     step_register = ModelStep(
#         name="TeamModelRegisterModel",
#         step_args=step_args,
#         depends_on=["TeamModel"]
#     )
    
    ############ BASELINE FUSION STEP ########################
    timestamp = str(time.time()).split('.')[0]
    output_prefix = f'{base_job_prefix}/fusion/baseline/outputs/{timestamp}'
    output_s3_uri = f's3://{default_bucket}/{output_prefix}'
    code_location = f's3://{default_bucket}/{base_job_prefix}/fusion/baseline/code'

    env_vars = {
        "SM_CHANNEL_VAL": "/opt/ml/processing/input/data/validation",
        "SM_MODEL_DIR": "/opt/ml/processing/model",
        "SM_OUTPUT_DIR": "/opt/ml/processing/output"
    }
    
    # Processing Script Arguments
    arguments = [
        "--samples-per-batch", chunk_size.to_string(), 
        "--batch-size", batch_size.to_string(),
        "--num-sensors", num_sensors.to_string()
    ]

    code = 'baseline_fusion.py'

    script_baseline_fusion = PyTorchProcessor(
        framework_version='1.13.1',
        py_version="py39",
        role=role,
        env=env_vars,
        instance_count=processing_instance_count,
        instance_type=fusion_instance_type,
        base_job_name = f"{base_job_prefix}-baseline-fusion",
        code_location=code_location,
        sagemaker_session=pipeline_session
    )

    step_args = script_baseline_fusion.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination=env_vars["SM_MODEL_DIR"],
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                destination=env_vars["SM_CHANNEL_VAL"],
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="output", source=env_vars["SM_OUTPUT_DIR"], destination = output_s3_uri),
        ],
        arguments=arguments,
        source_dir=code_dir,
        code=code
    )
    
    step_baseline_fusion = ProcessingStep(
        name="BaselineFusion",
        cache_config=cache_config,
        step_args = step_args,
        depends_on=["TrainModel"]
    )
    
    ############ RL-RFE FUSION STEP ########################
    timestamp = str(time.time()).split('.')[0]
    output_prefix = f'{base_job_prefix}/fusion/rl_rfe/outputs/{timestamp}'
    output_s3_uri = f's3://{default_bucket}/{output_prefix}'
    code_location = f's3://{default_bucket}/{base_job_prefix}/fusion/rl_rfe/code'
    
    env_vars = {
        "SM_CHANNEL_VAL": "/opt/ml/processing/input/data",
        "SM_OUTPUT_DIR": "/opt/ml/processing/output"
    }
    
    # Processing Script Arguments
    arguments = [
        "--num-sensors", num_sensors.to_string()
    ]

    code = 'rl_rfe_fusion.py'

    script_rl_rfe_fusion = PyTorchProcessor(
        framework_version='1.13.1',
        py_version="py39",
        role=role,
        env=env_vars,
        instance_count=processing_instance_count,
        instance_type=fusion_instance_type,
        base_job_name = f"{base_job_prefix}-rl-rfe-fusion",
        code_location=code_location,
        sagemaker_session=pipeline_session
    )

    step_args = script_rl_rfe_fusion.run(
        inputs=[
            ProcessingInput(
                source=step_baseline_fusion.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri,
                destination=env_vars["SM_CHANNEL_VAL"],
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="output", source=env_vars["SM_OUTPUT_DIR"], destination = output_s3_uri),
        ],
        arguments=arguments,
        source_dir=code_dir,
        code=code
    )
    
    step_rl_rfe_fusion = ProcessingStep(
        name="RLRFEFusion",
        cache_config=cache_config,
        step_args = step_args,
        depends_on=["BaselineFusion"]
    )
    
    ############ MODEL EVALUATION STEP ########################
    timestamp = str(time.time()).split('.')[0]
    output_prefix = f'{base_job_prefix}/evaluation/outputs/{timestamp}'
    output_s3_uri = f's3://{default_bucket}/{output_prefix}'
    code_location = f's3://{default_bucket}/{base_job_prefix}/evaluation/code'
    
    # Processing Script Arguments
    arguments = [
        "--samples-per-batch", chunk_size.to_string(), 
        "--batch-size", batch_size.to_string(),
        "--num-sensors", num_sensors.to_string()
    ]

    code = 'evaluation.py'
    
    env_vars = {
        "SM_CHANNEL_VAL": "/opt/ml/processing/input/data/validation",
        "SM_TEAM_MODEL_DIR": "/opt/ml/processing/model/team",
        "SM_BASELINE_MODEL_DIR": "/opt/ml/processing/model/baseline",
        "SM_RLRFE_MODEL_DIR": "/opt/ml/processing/model/rlrfe",
        "SM_BASE_MODEL_DIR": "/opt/ml/processing/model",
        "SM_OUTPUT_DIR": "/opt/ml/processing/output"
    }

    script_eval = PyTorchProcessor(
        framework_version='1.13.1',
        py_version="py39",
        role=role,
        env=env_vars,
        instance_count=processing_instance_count,
        instance_type=fusion_instance_type,
        base_job_name = f"{base_job_prefix}-evaluation",
        code_location=code_location,
        sagemaker_session=pipeline_session
    )
    
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination=env_vars["SM_TEAM_MODEL_DIR"],
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                destination=env_vars["SM_CHANNEL_VAL"],
            ),
            ProcessingInput(
                source=step_baseline_fusion.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri,
                destination=env_vars["SM_BASELINE_MODEL_DIR"],
            ),
            ProcessingInput(
                source=step_rl_rfe_fusion.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri,
                destination=env_vars["SM_RLRFE_MODEL_DIR"],
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="output", source=env_vars["SM_OUTPUT_DIR"], destination = output_s3_uri),
        ],
        arguments=arguments,
        source_dir=code_dir,
        code=code
    )
    
    step_eval = ProcessingStep(
        name="Evaluation",
        cache_config=cache_config,
        step_args = step_args,
        depends_on=["RLRFEFusion"]
    )
    
    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            fusion_instance_type,
            processing_instance_count,
            training_instance_type,
            training_instance_count,
            inference_instance_type,
            model_approval_status,
            chunk_size,
            num_sensors,
            batch_size,
            sample_size
        ],
        steps=[step_process, step_train, step_feature_contribution, step_feature_extraction, step_baseline_fusion, step_rl_rfe_fusion, step_eval],
        sagemaker_session=pipeline_session,
    )
    return pipeline
