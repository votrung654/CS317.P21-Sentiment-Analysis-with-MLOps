<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS317.P21 - MACHINE LEARNING SYSTEMS DEVELOPMENT AND OPERATIONS</b></h1>

# MLOps Getting Started

A comprehensive MLOps project implementing sentiment analysis with traditional machine learning and deep learning models, featuring complete containerization, deployment, and monitoring capabilities.

## TABLE OF CONTENTS
* [Course Information](#course-information)
* [Instructor](#instructor)
* [Team Members](#team-members)
* [Project Overview](#project-overview)
* [Features](#features)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Lab 1: ML Development & Experiment Tracking](#lab-1-ml-development--experiment-tracking)
* [Lab 2: Containerization & Deployment](#lab-2-containerization--deployment)
* [Lab 3: Monitoring & Observability](#lab-3-monitoring--observability)
* [Configuration & Customization](#configuration--customization)
* [Troubleshooting](#troubleshooting)
* [Requirements](#requirements)
* [Project Architecture](#project-architecture--implementation-phases)
* [Future Enhancements](#future-enhancements)

## COURSE INFORMATION
<a name="course-information"></a>
* **Course Name**: Machine Learning Systems Development and Operations
* **Course Code**: CS317
* **Class Code**: CS317.P21
* **Academic Year**: 2024-2025
* **Institution**: University of Information Technology (UIT), Vietnam National University - Ho Chi Minh City

## INSTRUCTOR
<a name="instructor"></a>
* MSc. **Đỗ Văn Tiến** - *tiendv@uit.edu.vn*

## TEAM MEMBERS
<a name="team-members"></a>
| STT    | MSSV          | Họ và Tên              | Github                                               | Email                   |
| ------ |:-------------:| ----------------------:|-----------------------------------------------------:|-------------------------:
| 1      | 22521571      | Võ Đình Trung          | https://github.com/votrung654                       |22521571@gm.uit.edu.vn  |
| 2      | 22520518      | Nguyễn Thanh Hùng      | https://github.com/nth4002                          |22520518@gm.uit.edu.vn  |
| 3      | 22520193      | Phan Thanh Đăng        | https://github.com/PTD504                           |22520193@gm.uit.edu.vn  |

## PROJECT OVERVIEW
<a name="project-overview"></a>

## 🎯 Overview

This project implements a full MLOps workflow for sentiment analysis, comparing traditional machine learning approaches (Logistic Regression, SVM, Naive Bayes) with deep learning models (LSTM). It includes data preprocessing, model training, hyperparameter optimization, distributed execution, experiment tracking, and model serving via a web API with comprehensive monitoring and observability.

## ✨ Features

### Core ML Capabilities
- **Data Preprocessing**: Text cleaning, lemmatization, and vectorization with TF-IDF
- **Multiple Models**: 
  - Traditional ML: Logistic Regression, SVM, Naive Bayes
  - Deep Learning: Bidirectional LSTM with Embedding
- **Explainability**: Basic feature importance visualization

### MLOps Components
- **Data Versioning**: DVC (Data Version Control) with MinIO for data pipeline management
- **Experiment Tracking**: MLflow for logging parameters, metrics, and artifacts
- **Hyperparameter Tuning**: Optuna for systematic optimization
- **Distributed Training**: Ray for parallel execution of pipeline components
- **Model Serving**: FastAPI for REST API endpoints with interactive documentation
- **Web Application**: Interactive UI for testing sentiment analysis models

### Production Features (Lab 3)
- **System Monitoring**: CPU, Memory, Disk usage tracking
- **API Monitoring**: Request rate, Error rate, Latency metrics
- **Model Monitoring**: Inference speed, Confidence scores
- **Structured Logging**: JSON format logging with multiple outputs
- **Alerting**: Alertmanager for anomaly detection and notifications
- **Containerization**: Full Docker deployment with monitoring stack

## 📊 Dataset

The project uses the IMDB Movie Reviews dataset containing 50,000 movie reviews labeled as positive or negative sentiment.

**Data Management:**
- **Source**: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 movie reviews with binary sentiment labels
- **Data Versioning**: Managed with DVC (Data Version Control) and MinIO storage
- **Splits**: Automatically divided into train/validation/test sets during pipeline execution

**Data Versioning with DVC:**
For comprehensive data versioning setup and management, we maintain a separate DVC repository. Please refer to our dedicated DVC guide: [📋 DVC & MinIO Setup Repository](https://github.com/nth4002/Sentiment_Analysis_with_MLOps) (contains detailed [dvc.md](https://github.com/nth4002/Sentiment_Analysis_with_MLOps/blob/main/dvc.md) documentation)

This implementation includes:
- Git-like versioning for large datasets
- Remote storage with MinIO (S3-compatible)
- Data pipeline reproducibility
- Collaborative data management across team members

## 📁 Project Structure

```
MLOps-getting-started/
├── app.py                          # FastAPI application with monitoring endpoints
├── pipeline.py                     # Main orchestration pipeline using Ray
├── traffic_generator.py            # Traffic generation script for testing
├── requirements.txt                # Project dependencies (all labs)
├── Dockerfile                      # Multi-stage container definition
├── docker-compose.yml              # Complete monitoring stack
├── docker-compose-hub.yml          # Deploy from Docker Hub image
├── README.md                       # This comprehensive documentation
├── EDA.ipynb                       # Exploratory Data Analysis notebook
├── deploy.sh                       # Server deployment script
├── push-to-dockerhub.sh            # Docker Hub publishing script
├── data/                           # Dataset storage
│   ├── IMDB-Dataset.csv           # Original IMDB dataset
│   ├── train.csv                  # Training split
│   ├── val.csv                    # Validation split
│   └── test.csv                   # Test split
├── src/                           # Source code modules
│   ├── data/
│   │   └── preprocessing.py       # Data preprocessing module
│   └── models/
│       ├── traditional_models.py  # Traditional ML models
│       └── deep_learning_models.py # LSTM and other DL models
├── models/                        # Saved models and artifacts
│   ├── logistic_regression.joblib # Trained logistic regression model
│   ├── logistic_regression_report.txt # Model evaluation report
│   ├── lstm_model.h5              # Trained LSTM model
│   ├── lstm_confusion_matrix.png  # LSTM evaluation visualization
│   └── tokenizer.pkl              # Text tokenizer for LSTM
├── static/                        # Static files for web UI (CSS, JS, images)
├── templates/                     # HTML templates for web interface
│   └── index.html                 # Main web interface
├── logs/                          # Application logs directory
├── prometheus/                    # Monitoring configuration
│   ├── prometheus.yml             # Prometheus configuration
│   ├── alertmanager.yml           # Alert routing rules
│   └── alert.rules.yml            # Alert definitions
├── grafana/                       # Visualization setup
│   └── provisioning/              # Grafana auto-provisioning
│       ├── datasources/           # Pre-configured data sources
│       └── dashboards/            # Custom dashboard definitions
├── loki/
│   └── loki-config.yml            # Log aggregation config
├── promtail/
│   └── promtail-config.yml        # Log collection config
└── mlruns/                        # MLflow experiment tracking data
```

## 🚀 Installation

### Prerequisites
- **Python**: 3.9+ (not compatible with Python 3.12 or higher due to certain dependencies)
- **Docker**: version 20.10.0 or later
- **Docker Compose**: version 2.0.0 or later
- **Memory**: 4GB+ RAM recommended for full monitoring stack

### Setup Steps

1. **Clone the repository**:
```bash
git clone https://github.com/PTD504/MLOps-getting-started.git
cd MLOps-getting-started
```

2. **Set up a virtual environment**:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:
```bash
# Install all dependencies for complete functionality
pip install -r requirements.txt
```

4. **Download the IMDB dataset**:
   - Download from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - Save as IMDB-Dataset.csv

5. **Set up Data Versioning (Optional but Recommended)**:
   - Follow our separate DVC repository setup: [MLOps DVC Setup](https://github.com/nth4002/Sentiment_Analysis_with_MLOps)
   - This provides Git-like versioning for large datasets using MinIO storage
   - Enables collaborative data management and reproducible experiments

## 📈 Lab 1: ML Development & Experiment Tracking

### Overview
Lab 1 focuses on building the core machine learning pipeline with experiment tracking, hyperparameter optimization, and data versioning capabilities.

### Data Versioning Setup (Prerequisites)

**Important**: For production-ready data versioning, we recommend setting up DVC with MinIO:

1. **Visit our DVC Setup Repository**: [MLOps DVC Setup](https://github.com/nth4002/Sentiment_Analysis_with_MLOps)
2. **Follow the detailed guide**: Contains complete DVC and MinIO configuration instructions
3. **Benefits include**:
   - **Data Reproducibility**: Exact dataset versions for each experiment
   - **Collaboration**: Team members can access the same data versions
   - **Storage Efficiency**: Large datasets stored separately from Git
   - **Pipeline Tracking**: Data lineage and transformation tracking

**Quick Setup (if not using separate DVC repo)**:
```bash
# If you want to set up DVC locally in this repo
dvc init --no-scm -f
# Configure your remote storage (MinIO/S3/GCS/etc.)
dvc remote add -d myremote s3://your-bucket
# Track your dataset
dvc add data/IMDB-Dataset.csv
git add data/IMDB-Dataset.csv.dvc data/.gitignore
git commit -m "Add dataset with DVC"
dvc push
```

### Data Validation with Great Expectations

Before training, the pipeline includes comprehensive data validation:

```bash
# Data validation is integrated into pipeline.py
# Checks include:
# - Schema validation (column types, names)
# - Data completeness (missing values detection)  
# - Data distribution checks
# - Text quality validation (length, encoding)
```

**Benefits of Data Validation:**
- Early detection of data quality issues
- Automated pipeline stopping on invalid data
- HTML reports for data quality assessment
- Reusable expectation suites

### Run the Full Pipeline

```bash
python pipeline.py
```

This will execute:
1. **Data Validation**: Quality checks and schema validation with Great Expectations
2. **Data Preprocessing**: Text cleaning and feature extraction using `src/data/preprocessing.py`
3. **Traditional ML Training**: Multiple models with Optuna hyperparameter tuning via traditional_models.py
4. **Deep Learning Training**: LSTM model with optimization using deep_learning_models.py
5. **Experiment Logging**: All parameters, metrics, and artifacts logged to MLflow

### View Experiment Tracking Results

```bash
mlflow ui
```
Access the MLflow dashboard at http://localhost:5000

### Key Features
- **Ray Integration**: Distributed execution for parallel model training
- **Optuna Optimization**: Systematic hyperparameter tuning with 100+ trials
- **MLflow Tracking**: Complete experiment lineage and artifact storage
- **Model Comparison**: Automated evaluation across multiple algorithms
- **Great Expectations**: Data quality validation and reporting

### Training Results

| Model | Accuracy | Features |
|-------|----------|----------|
| **Bidirectional LSTM** | **91.30%** | **Best accuracy, context understanding** |
| Logistic Regression | 89.36% | Fast inference, interpretable |
| SVM | 89.05% | Good with high-dimensional features |
| Naive Bayes | 86.99% | Fast, efficient for text data |

### Training Process Demo
- [Training Process Video](https://drive.google.com/file/d/1rPvdYF71s9emmPndpeG6CEJAPC7hnraU/view?usp=sharing)

## 🐳 Lab 2: Containerization & Deployment

### Overview
Lab 2 extends the ML models with web API deployment and containerization for production use.

### Local Development

Run API locally for development:
```bash
uvicorn app:app --host localhost --port 8000

# or 

python app.py
```

### Docker Deployment

#### Prerequisites for Docker Deployment
- Make sure you have trained models in the models directory:
  - `logistic_regression.joblib`
  - `lstm_model.h5`
  - `tokenizer.pkl`
- If you don't have models, run the pipeline first: `python pipeline.py`
- Static files in static directory (included in repository)
- HTML templates in templates directory (included in repository)

#### Deploy with Docker Compose

1. **Build and start the container**:
```bash
docker-compose up -d
```

2. **Access the services**:
   - **Web interface**: http://localhost:8000
   - **API documentation**: http://localhost:8000/docs
   - **Health check**: http://localhost:8000/info

#### API Endpoints

- `GET /`: Interactive web interface for testing predictions
- `POST /predict`: Main prediction endpoint (supports both traditional ML and LSTM)
- `GET /health`: Health check endpoint for monitoring
- `GET /info`: API information and version
- `GET /metrics`: Prometheus metrics endpoint
- `GET /docs`: Swagger UI documentation

#### Docker Hub Deployment

1. **Build and publish to Docker Hub**:
```bash
# Log in to Docker Hub
docker login

# Use the automated script
./push-to-dockerhub.sh your-username
```

2. **Run from Docker Hub**:
```bash
docker-compose -f docker-compose-hub.yml up -d
```

#### Server Deployment

Deploy to production server:
```bash
# Give execute permission to the deployment script
chmod +x deploy.sh

# Run deployment script with parameters
./deploy.sh username server-ip server-path dockerhub-username
```

### Key Features
- **FastAPI Framework**: High-performance async API with automatic documentation
- **Interactive Web Interface**: User-friendly sentiment analysis testing
- **Multi-model Support**: Switch between traditional ML and LSTM models
- **Docker Multi-stage Build**: Optimized container size and security
- **Production Ready**: Health checks, proper logging, and error handling

### Lab 2 Demo Videos
- **Docker Build & Deployment Tests**: [Video Collection](https://drive.google.com/drive/folders/1QZql71yOEhx4iyF9JAs8mpA3C-xdXVwe?usp=sharing)
  - `access_the_api_after_build`
  - `BuildAndStartContainer`
  - `publish_to_docker_hub`
  - `run_the_service_from_image_n_docker_hub`
- **Server Deployment Demo**: [Production Deploy Video](https://drive.google.com/file/d/1vAXwRElNjsoeqkng31pU12-9BI4JpP9t/view?usp=drive_link)

## 📊 Lab 3: Monitoring & Observability

### Overview
Lab 3 implements comprehensive monitoring and logging solution for the sentiment analysis API with complete observability stack.

### Lab 3 Assignment Requirements

With the API built in Lab 2, this project implements complete Monitoring and Logging services with the following requirements:

### 1. Server Resource Monitoring ✅
- **CPU usage**: Monitored via Node Exporter
- **RAM usage**: Memory utilization tracking
- **Disk space & Disk IO**: Storage metrics and I/O operations
- **Network IO**: Total transmitted/received data tracking
- **GPU usage**: Optional monitoring (implemented when GPU available)

### 2. API Monitoring ✅
- **Request per second**: Real-time RPS tracking
- **Error rate**: HTTP error status monitoring with alerts
- **Latency**: Request/response time measurement with percentiles

### 3. Model Monitoring ✅
- **Inference speed**: CPU/GPU execution time tracking
- **Confidence score**: Model prediction confidence with alerts

### 4. Comprehensive Logging ✅
- **syslog**: System-level logs for infrastructure issues
- **stdout**: Console output streams
- **stderr**: Error traceback logging
- **logfile**: Application-specific log files

### 5. Alerting System ✅
- **Error rate alerts**: Triggers when error rate > 50%
- **Low confidence alerts**: Activates when confidence < 0.6
- **Resource alerts**: CPU, memory, disk space thresholds
- **Customizable thresholds**: Easily configurable alert conditions

### Technology Stack

- **Monitoring**: Prometheus + Grafana
- **Logging**: Loki + Promtail
- **Alerting**: Alertmanager
- **API Instrumentation**: prometheus-fastapi-instrumentator
- **Infrastructure Metrics**: Node Exporter
- **Containerization**: Docker + Docker Compose

### Installation and Setup

1. **Start the complete monitoring stack**:
```bash
# Stop any conflicting containers
docker-compose down -v

# Build and start all services
docker-compose up --build -d
```

2. **Verify all services are running**:
```bash
# Check container status
docker-compose ps

# Test API endpoints
curl http://localhost:8000/info
curl http://localhost:8000/health
curl http://localhost:8000/metrics

# Test monitoring services
curl http://localhost:9090/targets  # Prometheus targets
```

### Service Access URLs

| Service | URL | Credentials | Description |
|---------|-----|-------------|-------------|
| **API** | http://localhost:8000 | - | Main application interface |
| **API Docs** | http://localhost:8000/docs | - | Interactive API documentation |
| **Prometheus** | http://localhost:9090 | - | Metrics collection and querying |
| **Grafana** | http://localhost:3000 | admin/admin | Dashboards and visualization |
| **Alertmanager** | http://localhost:9093 | - | Alert management and routing |
| **Node Exporter** | http://localhost:9100 | - | System metrics collection |

### Testing and Traffic Generation

1. **Generate test traffic for monitoring**:
```bash
python traffic_generator.py
```

This script generates:
- **Phase 1**: Normal traffic (120 seconds at 3 RPS)
- **Phase 2**: Error generation (30 seconds) 
- **Phase 3**: High traffic load (60 seconds at 5 RPS)

2. **Monitor in Grafana**:
   - Login to http://localhost:3000 with admin/admin
   - Import pre-built dashboards or create custom ones
   - Observe real-time metrics during traffic generation

3. **Test alerting system**:
   - Script automatically generates errors to trigger alerts
   - Check Alertmanager UI at http://localhost:9093 for active alerts
   - Verify alert notifications (if configured)

### Monitoring Metrics

#### System Metrics (Node Exporter)
- **CPU**: Usage percentage, load average
- **Memory**: Total, used, available, swap usage
- **Disk**: Usage percentage, I/O operations, read/write rates
- **Network**: Traffic in/out, packet rates, errors

#### API Metrics (Prometheus + FastAPI Instrumentator)
- **Request Rate**: Requests per second by endpoint and method
- **Error Rate**: HTTP error percentage by status code
- **Response Latency**: P50, P95, P99 percentiles
- **Active Connections**: Concurrent request count
- **Request Duration**: Histogram of response times

#### Model-Specific Metrics
- **Prediction Latency**: Inference time by model type (traditional vs LSTM)
- **Confidence Scores**: Distribution of prediction confidence
- **Model Usage**: Request count per model type
- **Prediction Results**: Sentiment distribution (positive/negative)

### Logging Architecture

**Multi-layer logging system**:
- **Console Output**: Development debugging and real-time monitoring
- **File Logging**: Persistent storage in `logs/` directory with rotation
- **Loki Integration**: Centralized log aggregation and forwarding
- **Structured Format**: JSON logs for machine processing
- **Log Levels**: Configurable verbosity (DEBUG, INFO, WARNING, ERROR)

### Alerting Rules

Alerts are automatically triggered when:
- **High Error Rate**: >50% errors sustained for 5 minutes
- **Slow Response Time**: API response time >5 seconds consistently
- **Low Model Confidence**: Confidence scores <0.6 for multiple predictions
- **Resource Exhaustion**: System resource usage >80% (CPU/Memory)
- **Service Health**: Container restarts or failures
- **High Request Volume**: Unusual traffic spikes

### Lab 3 Demo Video
**Demo Video**: [Lab 3 Monitoring & Logging Demo](https://drive.google.com/file/d/1kz0grRHgfGDE0eng2kFirgOmrQ4-Fk5S/view?usp=sharing)

**Video Content Verification:**
- ✅ Complete dashboard walkthrough showing all required metrics
- ✅ traffic_generator.py execution with real-time dashboard updates
- ✅ Error rate simulation exceeding 50% threshold  
- ✅ `HighErrorRate` alert progression from "inactive" to "FIRING"
- ✅ Multi-source log capture (syslog, stdout, stderr) demonstration
- ✅ Alertmanager UI showing active alerts
- ✅ All service containers running verification

## 🔧 Configuration & Customization

### Alert Rules Configuration

Edit alert.rules.yml:

```yaml
groups:
  - name: api_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[2m]) > 0.5  # >50%
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          
      - alert: LowConfidenceScore  
        expr: model_confidence_score < 0.6
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Low model confidence detected"
```

### Custom Metrics Implementation

In app.py, custom metrics are implemented:

```python
# Custom model monitoring metrics
model_inference_duration = Histogram('model_inference_duration_seconds', 'Model inference time')
model_confidence_score = Gauge('model_confidence_score', 'Model prediction confidence')
model_predictions_total = Counter('model_predictions_total', 'Total model predictions')
```

### Log Configuration

Promtail configuration in promtail-config.yml captures logs from multiple sources:

```yaml
scrape_configs:
  - job_name: containers
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
        
  - job_name: syslog
    static_configs:
      - targets:
          - localhost
        labels:
          job: syslog
          __path__: /var/log/syslog
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

1. **Check container logs**:
```bash
# View logs for specific service
docker-compose logs <service-name>
# Available services: sentiment_app_container, prometheus, grafana, alertmanager, node_exporter

# Follow logs in real-time
docker-compose logs -f <service-name>
```

2. **Restart services**:
```bash
# Restart entire stack
docker-compose restart

# Restart specific service
docker-compose restart alertmanager
```

3. **Verify endpoints and connectivity**:
```bash
# Check API health and metrics
curl http://localhost:8000/health
curl http://localhost:8000/metrics

# Verify Prometheus targets
curl http://localhost:9090/api/v1/targets

# Test model prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great movie!", "model_type": "traditional"}'
```

4. **Debug container issues**:
```bash
# Check container status
docker ps

# Inspect container configuration
docker inspect sentiment_app_container

# Access container shell for debugging
docker exec -it sentiment_app_container /bin/bash
```

### Port Conflicts

```bash
# Stop conflicting services
docker-compose down -v

# Check if ports are in use (Linux/Mac)
sudo lsof -i :3000,8000,9090,9093,9100,3100

# Check if ports are in use (Windows)
netstat -an | findstr "3000 8000 9090 9093 9100 3100"
```

## 📋 Requirements

### Core Dependencies

```txt
# Core ML Dependencies
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.6.1
tensorflow==2.16.1
nltk==3.8.1
joblib==1.3.2

# MLOps & Experiment Tracking (Lab 1)
mlflow==2.21.3
optuna==3.5.0
ray==2.7.0
great-expectations==0.18.12

# Web API & Deployment (Lab 2)
fastapi==0.110.0
uvicorn==0.27.1
jinja2==3.1.3
python-multipart==0.0.9

# Monitoring & Observability (Lab 3)
prometheus-fastapi-instrumentator==6.1.0
prometheus-client==0.19.0
psutil==5.9.8
structlog==23.2.0
```

Complete dependencies list available in requirements.txt.

## 🏗️ Project Architecture & Implementation Phases

This MLOps project is structured into three distinct phases, each building upon the previous to create a comprehensive production-ready machine learning system:

### Phase 1: ML Development & Experiment Tracking 🔬
**Objective**: Establish a reproducible, automated, and version-controlled model training pipeline.

**Key Components:**
- **Data Versioning**: DVC integration with MinIO for Git-like data management
- **Data Validation**: Great Expectations for data quality assurance
- **Distributed Training**: Ray framework for parallel model training and hyperparameter optimization
- **Experiment Tracking**: MLflow server for comprehensive experiment logging and model registry
- **Model Development**: Comparison of traditional ML (Logistic Regression, SVM, Naive Bayes) vs Deep Learning (Bidirectional LSTM)
- **Hyperparameter Optimization**: Optuna for systematic parameter tuning

### Phase 2: Containerization & Deployment 🚀
**Objective**: Transform trained models into a production-ready API service with proper containerization.

**Key Components:**
- **API Development**: FastAPI with automatic documentation and interactive web interface
- **Containerization**: Multi-stage Docker build for optimized production deployment
- **Model Serving**: REST endpoints for sentiment prediction with multiple model support
- **Web Interface**: User-friendly HTML interface for testing and demonstration
- **CI/CD Pipeline**: Automated Docker Hub publishing and server deployment scripts
- **Health Monitoring**: API health checks and basic performance metrics

### Phase 3: Comprehensive Monitoring & Observability 📊
**Objective**: Implement enterprise-grade monitoring, logging, and alerting for production systems.

**Key Components:**
- **Infrastructure Monitoring**: Prometheus + Node Exporter for system resource tracking
- **Application Monitoring**: API performance metrics with custom instrumentations
- **Visualization**: Grafana dashboards for real-time monitoring and analytics
- **Centralized Logging**: Loki + Promtail for multi-source log aggregation
- **Alerting System**: Alertmanager for automated notifications and threshold-based alerts
- **Model Monitoring**: Inference performance and prediction confidence tracking

## 🔧 Technology Stack by Phase

| Phase | Category | Technologies |
|-------|----------|-------------|
| **Phase 1** | Data & ML | DVC, MinIO, MLflow, Ray, Optuna, Scikit-learn, TensorFlow, Great Expectations |
| | Storage & Versioning | Git, Pandas, Jupyter Notebooks |
| **Phase 2** | API & Web | FastAPI, Uvicorn, Pydantic, Jinja2, HTML/CSS |
| | Containerization | Docker, Docker Compose, Docker Hub |
| **Phase 3** | Monitoring | Prometheus, Grafana, Node Exporter, AlertManager |
| | Logging & Alerting | Loki, Promtail, Structured Logging (JSON) |

## 🏛️ System Architecture

```
┌─────────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   FastAPI App      │────│  Prometheus  │────│    Grafana      │
│ (sentiment_app_     │    │  (port 9090) │    │   (port 3000)   │
│  container:8000)    │    │              │    │                 │
└─────────────────────┘    └──────────────┘    └─────────────────┘
         │                       │                    │
         │                       │                    │
    ┌─────────┐            ┌─────────────┐      ┌──────────┐
    │  Logs   │────────────│  Promtail   │──────│   Loki   │
    │ (multi- │            │             │      │(port 3100)│
    │ source) │            │             │      │          │
    └─────────┘            └─────────────┘      └──────────┘
         │                                            │
    ┌─────────────┐                                   │
    │Node Exporter│───────────────────────────────────┘
    │(port 9100)  │
    └─────────────┘
         │
    ┌─────────────┐
    │Alertmanager │
    │(port 9093)  │
    └─────────────┘
```

## 🚀 Future Enhancements

### Technical Improvements
- **GPU Monitoring**: Extend Node Exporter with nvidia-docker integration for GPU resource tracking
- **Custom Dashboards**: Domain-specific visualization panels for sentiment analysis metrics
- **Advanced Alerting**: Integration with Slack, Email, or PagerDuty for production notifications
- **Log Analysis**: Automated log pattern recognition and anomaly detection
- **Model Drift Detection**: Statistical monitoring for model performance degradation over time
- **Distributed Tracing**: Request flow tracking across microservices

### MLOps Pipeline Enhancements  
- **A/B Testing**: Framework for comparing model performance in production
- **Advanced Security**: Authentication, authorization, and security scanning
- **Multi-environment Deployment**: Development, staging, and production environments
- **Canary Deployments**: Gradual rollout strategies for model updates
- **Feature Store**: Centralized feature management and serving
- **Data Drift Detection**: Automated monitoring for input data distribution changes
- **Automated Retraining**: CI/CD pipelines for model retraining on data drift

### Scalability & Performance
- **Kubernetes Deployment**: Container orchestration for high availability
- **Auto-scaling**: Dynamic resource allocation based on traffic patterns  
- **Model Serving Optimization**: TensorFlow Serving or Triton for high-performance inference
- **Caching Layer**: Redis integration for improved response times
- **Load Balancing**: Multiple model instance management
- **Edge Deployment**: Model serving at edge locations for reduced latency
