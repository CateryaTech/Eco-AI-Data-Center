# 🚀 Deployment Guide — Eco AI Data Center

> Enterprise deployment on Kubernetes with full monitoring, quantum simulation,
> background tasks, and ethical AI guardrails.

**CateryaTech** | [cateryatech@proton.me](mailto:cateryatech@proton.me) | [github.com/cateryatech](https://github.com/cateryatech)

---

## 📋 Table of Contents

1. [Local Development](#1-local-development)
2. [Docker](#2-docker)
3. [Kubernetes — Full Stack](#3-kubernetes--full-stack)
4. [Environment Variables](#4-environment-variables)
5. [Celery Workers](#5-celery-workers)
6. [Prometheus Integration](#6-prometheus-integration)
7. [InfluxDB Integration](#7-influxdb-integration)
8. [Scaling & Resource Recommendations](#8-scaling--resource-recommendations)
9. [Health Checks & Monitoring](#9-health-checks--monitoring)
10. [Security Hardening](#10-security-hardening)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Local Development

### Prerequisites

- Python **3.12+**
- pip or conda
- Redis (for Celery, optional)

### Quick Start

```bash
git clone https://github.com/cateryatech/Eco-AI-Data-Center.git
cd Eco-AI-Data-Center

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Install core dependencies
pip install -r requirements.txt

# Optional: install full stack
pip install torch pennylane celery redis boto3 dask[dataframe] \
            influxdb-client requests

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run app
streamlit run app.py
```

Open: **http://localhost:8501**

### Start Celery Worker (optional)

```bash
# In a separate terminal (Redis must be running)
celery -A workers.tasks worker --loglevel=info --concurrency=4

# Celery Beat scheduler (for periodic tasks)
celery -A workers.tasks beat --loglevel=info
```

---

## 2. Docker

### Dockerfile

```dockerfile
FROM python:3.12-slim

LABEL maintainer="cateryatech@proton.me"
LABEL org.opencontainers.image.title="Eco AI Data Center"
LABEL org.opencontainers.image.url="https://github.com/cateryatech/Eco-AI-Data-Center"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        celery redis pennylane dask[dataframe] boto3 influxdb-client requests

# Copy source
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
```

### docker-compose.yml (full stack)

```yaml
version: "3.9"

services:
  # ── Streamlit App ────────────────────────────────────────────────────────
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
      - CATERYA_COS_THRESHOLD=0.7
      - ALERT_SLACK_WEBHOOK=${ALERT_SLACK_WEBHOOK}
      - ALERT_EMAIL_TO=${ALERT_EMAIL_TO}
    depends_on:
      - redis
      - influxdb
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ── Celery Worker ────────────────────────────────────────────────────────
  worker:
    build: .
    command: celery -A workers.tasks worker --loglevel=info --concurrency=4
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      - redis
    restart: unless-stopped

  # ── Celery Beat ──────────────────────────────────────────────────────────
  beat:
    build: .
    command: celery -A workers.tasks beat --loglevel=info
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis
    restart: unless-stopped

  # ── Redis (Celery broker) ────────────────────────────────────────────────
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  # ── InfluxDB (time-series store) ─────────────────────────────────────────
  influxdb:
    image: influxdb:2.7-alpine
    ports:
      - "8086:8086"
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=changeme123
      - DOCKER_INFLUXDB_INIT_ORG=cateryatech
      - DOCKER_INFLUXDB_INIT_BUCKET=datacenter
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=eco-ai-influx-token
    volumes:
      - influxdb_data:/var/lib/influxdb2
    restart: unless-stopped

  # ── Prometheus ───────────────────────────────────────────────────────────
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./deployment/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

volumes:
  redis_data:
  influxdb_data:
  prometheus_data:
```

```bash
# Build and run
docker compose up -d

# Check logs
docker compose logs -f app
docker compose logs -f worker
```

---

## 3. Kubernetes — Full Stack

### Prerequisites

- `kubectl` configured
- Kubernetes cluster (EKS / GKE / k3s / minikube)
- `helm` (optional, for ingress)

### Namespace

```bash
kubectl create namespace eco-ai-datacenter
```

### Secrets

```bash
kubectl create secret generic eco-ai-secrets \
  --namespace=eco-ai-datacenter \
  --from-literal=ALERT_SLACK_WEBHOOK="https://hooks.slack.com/your-webhook" \
  --from-literal=ALERT_EMAIL_TO="ops@yourcompany.com" \
  --from-literal=ALERT_SMTP_HOST="smtp.gmail.com" \
  --from-literal=ALERT_SMTP_USER="your@email.com" \
  --from-literal=ALERT_SMTP_PASSWORD="your-app-password" \
  --from-literal=INFLUXDB_TOKEN="eco-ai-influx-token" \
  --from-literal=AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE" \
  --from-literal=AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
```

### ConfigMap

```yaml
# deployment/k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: eco-ai-config
  namespace: eco-ai-datacenter
data:
  CATERYA_COS_THRESHOLD: "0.7"
  CATERYA_SWARM_THRESHOLD: "0.7"
  CATERYA_MODEL_ID: "eco-ai-k8s-prod"
  CATERYA_VERBOSE: "false"
  CELERY_BROKER_URL: "redis://redis-service:6379/0"
  CELERY_RESULT_BACKEND: "redis://redis-service:6379/1"
  INFLUXDB_URL: "http://influxdb-service:8086"
  INFLUXDB_ORG: "cateryatech"
  INFLUXDB_BUCKET: "datacenter"
  AWS_DEFAULT_REGION: "ap-southeast-1"
```

```bash
kubectl apply -f deployment/k8s/configmap.yaml
```

### Redis Deployment

```yaml
# deployment/k8s/redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: eco-ai-datacenter
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports:
            - containerPort: 6379
          args: ["redis-server", "--appendonly", "yes"]
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          volumeMounts:
            - mountPath: /data
              name: redis-storage
      volumes:
        - name: redis-storage
          persistentVolumeClaim:
            claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: eco-ai-datacenter
spec:
  selector:
    app: redis
  ports:
    - port: 6379
      targetPort: 6379
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: eco-ai-datacenter
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 2Gi
```

### Streamlit App Deployment

```yaml
# deployment/k8s/app.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: eco-ai-app
  namespace: eco-ai-datacenter
  labels:
    app: eco-ai-app
    version: "1.0.0"
spec:
  replicas: 2
  selector:
    matchLabels:
      app: eco-ai-app
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: eco-ai-app
    spec:
      containers:
        - name: eco-ai-app
          image: cateryatech/eco-ai-datacenter:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8501
              name: http
          envFrom:
            - configMapRef:
                name: eco-ai-config
            - secretRef:
                name: eco-ai-secrets
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
          livenessProbe:
            httpGet:
              path: /_stcore/health
              port: 8501
            initialDelaySeconds: 30
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /_stcore/health
              port: 8501
            initialDelaySeconds: 15
            periodSeconds: 10
          volumeMounts:
            - mountPath: /app/data
              name: data-volume
      volumes:
        - name: data-volume
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: eco-ai-service
  namespace: eco-ai-datacenter
spec:
  selector:
    app: eco-ai-app
  ports:
    - name: http
      port: 80
      targetPort: 8501
  type: ClusterIP
```

### Celery Worker Deployment

```yaml
# deployment/k8s/celery-worker.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-worker
  namespace: eco-ai-datacenter
spec:
  replicas: 3
  selector:
    matchLabels:
      app: celery-worker
  template:
    metadata:
      labels:
        app: celery-worker
    spec:
      containers:
        - name: celery-worker
          image: cateryatech/eco-ai-datacenter:latest
          command:
            - celery
            - -A
            - workers.tasks
            - worker
            - --loglevel=info
            - --concurrency=4
            - -Q
            - celery,monitoring,evaluation
          envFrom:
            - configMapRef:
                name: eco-ai-config
            - secretRef:
                name: eco-ai-secrets
          resources:
            requests:
              memory: "256Mi"
              cpu: "200m"
            limits:
              memory: "1Gi"
              cpu: "800m"
          livenessProbe:
            exec:
              command:
                - celery
                - -A
                - workers.tasks
                - inspect
                - ping
                - -d
                - celery@$HOSTNAME
            initialDelaySeconds: 60
            periodSeconds: 60
            timeoutSeconds: 20
---
# Celery Beat (single replica — do NOT scale beyond 1)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-beat
  namespace: eco-ai-datacenter
spec:
  replicas: 1      # ← ALWAYS 1. Multiple beats = duplicate tasks.
  selector:
    matchLabels:
      app: celery-beat
  template:
    metadata:
      labels:
        app: celery-beat
    spec:
      containers:
        - name: celery-beat
          image: cateryatech/eco-ai-datacenter:latest
          command:
            - celery
            - -A
            - workers.tasks
            - beat
            - --loglevel=info
          envFrom:
            - configMapRef:
                name: eco-ai-config
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "200m"
```

### Horizontal Pod Autoscaler

```yaml
# deployment/k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: eco-ai-app-hpa
  namespace: eco-ai-datacenter
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: eco-ai-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: celery-worker-hpa
  namespace: eco-ai-datacenter
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: celery-worker
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 65
```

### Ingress

```yaml
# deployment/k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: eco-ai-ingress
  namespace: eco-ai-datacenter
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    # Required for Streamlit WebSocket
    nginx.ingress.kubernetes.io/proxy-http-version: "1.1"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
spec:
  tls:
    - hosts:
        - eco-ai.cateryatech.com
      secretName: eco-ai-tls
  rules:
    - host: eco-ai.cateryatech.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: eco-ai-service
                port:
                  number: 80
```

### Deploy Everything

```bash
# Apply all manifests
kubectl apply -f deployment/k8s/

# Watch pods come up
kubectl get pods -n eco-ai-datacenter -w

# Check app logs
kubectl logs -f deployment/eco-ai-app -n eco-ai-datacenter

# Check worker logs
kubectl logs -f deployment/celery-worker -n eco-ai-datacenter

# Port-forward for local testing
kubectl port-forward service/eco-ai-service 8501:80 -n eco-ai-datacenter
```

---

## 4. Environment Variables

| Variable | Default | Required | Description |
|---|---|---|---|
| `CATERYA_COS_THRESHOLD` | `0.7` | No | COS ethical guardrail |
| `CATERYA_MODEL_ID` | `eco-ai-optimizer` | No | Provenance chain model ID |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | For Celery | Redis broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/1` | For Celery | Redis results |
| `INFLUXDB_URL` | `http://localhost:8086` | For InfluxDB | InfluxDB endpoint |
| `INFLUXDB_TOKEN` | — | For InfluxDB | InfluxDB auth token |
| `INFLUXDB_ORG` | `cateryatech` | For InfluxDB | Organisation name |
| `INFLUXDB_BUCKET` | `datacenter` | For InfluxDB | Target bucket |
| `PROMETHEUS_URL` | `http://localhost:9090` | For Prometheus | Prometheus endpoint |
| `ALERT_SLACK_WEBHOOK` | — | For Slack alerts | Webhook URL |
| `ALERT_EMAIL_TO` | — | For email alerts | Recipient address |
| `ALERT_SMTP_HOST` | — | For email alerts | SMTP server |
| `ALERT_SMTP_USER` | — | For email alerts | SMTP username |
| `ALERT_SMTP_PASSWORD` | — | For email alerts | SMTP password |
| `AWS_ACCESS_KEY_ID` | — | For AWS scaling | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | — | For AWS scaling | AWS secret |
| `AWS_DEFAULT_REGION` | `ap-southeast-1` | No | AWS region |

---

## 5. Celery Workers

### Start Workers Manually

```bash
# General worker
celery -A workers.tasks worker --loglevel=info --concurrency=4

# Beat scheduler (once only)
celery -A workers.tasks beat --loglevel=info

# Flower monitoring dashboard
pip install flower
celery -A workers.tasks flower --port=5555
# Open http://localhost:5555
```

### Task Queue Configuration

The following beat schedule is configured in `workers/tasks.py`:

| Task | Interval | Description |
|---|---|---|
| `run_monitor_poll` | Every 10s | Sensor poll cycle |
| `run_cos_evaluation` | Every 5 min | Full COS evaluation |
| `run_quantum_thermal` | Every 30 min | Quantum thermal simulation |

---

## 6. Prometheus Integration

### prometheus.yml scrape config

```yaml
# deployment/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "eco-ai-datacenter"
    static_configs:
      - targets:
          - "eco-ai-service:8501"
    metrics_path: /metrics

  - job_name: "celery-workers"
    static_configs:
      - targets:
          - "celery-worker:9090"
```

### Expose custom metrics (optional)

Add this to `app.py` to expose Prometheus metrics:

```python
# pip install prometheus-client
from prometheus_client import Gauge, start_http_server

pue_gauge = Gauge("datacenter_pue", "Power Usage Effectiveness")
cos_gauge = Gauge("caterya_cos_score", "CATERYA Open Score")

# In your evaluation loop:
pue_gauge.set(pue_result["pue"])
cos_gauge.set(cos.composite)

start_http_server(9090)
```

---

## 7. InfluxDB Integration

### Configure in .env

```bash
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=eco-ai-influx-token
INFLUXDB_ORG=cateryatech
INFLUXDB_BUCKET=datacenter
```

### Switch backend in app

In the sidebar or environment:

```bash
MONITOR_BACKEND=influxdb
```

### Write data to InfluxDB programmatically

```python
from real_time_monitor import InfluxDBBackend, MockSensorBackend

backend = InfluxDBBackend(
    url="http://localhost:8086",
    token="eco-ai-influx-token",
    org="cateryatech",
    bucket="datacenter",
)

# Read from mock sensor and write to InfluxDB
mock = MockSensorBackend()
readings = mock.read_all()
backend.write(readings)
```

---

## 8. Scaling & Resource Recommendations

### Minimum Production Setup

| Component | Replicas | CPU | Memory |
|---|---|---|---|
| Streamlit App | 2 | 500m | 1Gi |
| Celery Worker | 2 | 500m | 512Mi |
| Celery Beat | 1 | 100m | 128Mi |
| Redis | 1 | 200m | 256Mi |
| InfluxDB | 1 | 500m | 1Gi |

### Large Enterprise Setup (10k+ metrics/s)

| Component | Replicas | CPU | Memory |
|---|---|---|---|
| Streamlit App | 5 | 2000m | 4Gi |
| Celery Worker | 10 | 2000m | 2Gi |
| Celery Beat | 1 | 200m | 256Mi |
| Redis Cluster | 3 | 1000m | 2Gi |
| InfluxDB | 3 | 2000m | 8Gi |

---

## 9. Health Checks & Monitoring

```bash
# App health
curl http://localhost:8501/_stcore/health

# Celery worker health
celery -A workers.tasks inspect ping

# Redis health
redis-cli ping

# InfluxDB health
curl http://localhost:8086/health

# Kubernetes readiness
kubectl get pods -n eco-ai-datacenter
kubectl describe pod <pod-name> -n eco-ai-datacenter

# View all events
kubectl get events -n eco-ai-datacenter --sort-by='.lastTimestamp'
```

---

## 10. Security Hardening

### Network Policies

```yaml
# deployment/k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: eco-ai-network-policy
  namespace: eco-ai-datacenter
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: eco-ai-datacenter
    - ports:
        - port: 8501
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: eco-ai-datacenter
    - ports:
        - port: 443    # HTTPS
        - port: 587    # SMTP
```

### Pod Security

```yaml
# In pod spec:
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: false
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
```

---

## 11. Troubleshooting

### App won't start

```bash
# Check logs
kubectl logs -f deployment/eco-ai-app -n eco-ai-datacenter

# Exec into pod
kubectl exec -it deployment/eco-ai-app -n eco-ai-datacenter -- bash

# Check env vars
kubectl exec deployment/eco-ai-app -n eco-ai-datacenter -- env | grep CATERYA
```

### Celery tasks not running

```bash
# Check worker is alive
celery -A workers.tasks inspect ping

# Check active tasks
celery -A workers.tasks inspect active

# Check beat is running
celery -A workers.tasks inspect scheduled
```

### COS score always low

- Check data quality — missing values penalise `information_score`
- Check for heavily skewed distributions — penalises `symmetry_score`
- Lower `CATERYA_COS_THRESHOLD` temporarily while debugging
- Run `pytest tests/ -v -k TestCATERYAEvaluator` to isolate the issue

### PennyLane not available

```bash
pip install pennylane
# The system falls back to classical numpy simulation automatically
```

### InfluxDB connection refused

```bash
# Check InfluxDB is running
curl http://localhost:8086/health

# Check token is correct
influx auth list --token eco-ai-influx-token
```

---

## 📬 Support

**Ary HH** — *CateryaTech*  
📧 [cateryatech@proton.me](mailto:cateryatech@proton.me)  
🐙 [github.com/cateryatech](https://github.com/cateryatech)

---

*Built with 🌿 ethical AI principles. CATERYA-certified.*

---

## Monetization & Billing Deployment

### Stripe Setup

```bash
# 1. Create products in Stripe dashboard
# Products: Eco AI Starter / Growth / Enterprise / Government

# 2. Get price IDs and add to .env
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PRICE_STARTER=price_xxx
STRIPE_PRICE_GROWTH=price_yyy
STRIPE_PRICE_ENTERPRISE=price_zzz
STRIPE_PRICE_GOVERNMENT=price_www

# 3. Setup webhook endpoint in Stripe dashboard
# URL: https://your-domain.com/api/v1/billing/webhook
# Events: customer.subscription.updated, customer.subscription.deleted,
#         invoice.payment_succeeded, invoice.payment_failed
STRIPE_WEBHOOK_SECRET=whsec_...

# 4. Configure Stripe Meter events (for pay-per-use)
# In Stripe dashboard: Billing → Meters
# Create meters: simulation_run, api_call, compliance_scan, esg_badge_mint, etc.
```

### Analytics Forwarding

```bash
# Mixpanel
MIXPANEL_TOKEN=your-token

# Google Analytics 4 (Measurement Protocol)
GA_MEASUREMENT_ID=G-XXXXXXXXXX
GA_API_SECRET=your-secret
# Note: GA4 forwarding requires server-side GA4 property
```

### Multi-Tenant Database

For production multi-tenant deployments, replace in-memory store with PostgreSQL:

```python
# In monetization/analytics.py — extend UsageTracker with DB backend
# In monetization/billing.py — extend SubscriptionManager with DB backend
# Schema: tenants, subscriptions, usage_events, invoices
```

### Heroku Deployment

```bash
# Create Heroku app
heroku create eco-ai-datacenter-prod

# Add buildpacks
heroku buildpacks:add heroku/python

# Set environment variables
heroku config:set APP_ENV=production
heroku config:set STRIPE_SECRET_KEY=sk_live_...
heroku config:set MASTER_ENCRYPTION_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
heroku config:set JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Add Redis for Celery
heroku addons:create heroku-redis:mini

# Deploy
git push heroku main

# Scale workers
heroku ps:scale web=1 worker=2
```

### AWS Deployment (ECS + Fargate)

```yaml
# task-definition.json (excerpt)
{
  "containerDefinitions": [{
    "name": "eco-ai-datacenter",
    "image": "your-ecr-repo/eco-ai-datacenter:latest",
    "portMappings": [{"containerPort": 8000}, {"containerPort": 8501}],
    "environment": [
      {"name": "APP_ENV", "value": "production"},
      {"name": "STRIPE_SECRET_KEY", "valueFrom": "arn:aws:ssm:...STRIPE_SECRET_KEY"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {"awslogs-group": "/eco-ai/prod", "awslogs-region": "ap-southeast-1"}
    }
  }]
}
```

```bash
# Build and push
docker build -t eco-ai-datacenter .
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
docker tag eco-ai-datacenter:latest $ECR_URI/eco-ai-datacenter:latest
docker push $ECR_URI/eco-ai-datacenter:latest

# Deploy to ECS
aws ecs update-service --cluster eco-ai-prod --service eco-ai-datacenter --force-new-deployment
```

---

## Ethical Certification in CI/CD

Add CATERYA gate to your GitHub Actions / GitLab CI pipeline:

```yaml
# .github/workflows/caterya-gate.yml
name: CATERYA Ethics Gate

on: [push, pull_request]

jobs:
  caterya-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.12'}
      - run: pip install -r requirements.txt
      - name: Run full test suite
        run: |
          python3 tests/test_monetization_e2e.py
          python3 tests/test_blockchain_agentic.py
          python3 tests/test_enterprise_security_v2.py
      - name: CATERYA COS Gate
        run: |
          python3 -c "
          import sys; sys.path.insert(0,'.')
          import numpy as np, pandas as pd
          from caterya_framework.evaluator import CATERYAEvaluator
          from caterya_framework.swarm import EthicsSwarm
          rng = np.random.default_rng(42)
          df = pd.DataFrame({'power_kw': rng.normal(1200,100,80).clip(800,2000),
              'it_power_kw': rng.normal(700,60,80).clip(400,1200),
              'water_usage_litres': rng.normal(500,40,80).clip(200,900),
              'carbon_intensity_kg_per_kwh': rng.normal(0.28,0.07,80).clip(0.05,0.6)})
          cos = CATERYAEvaluator().score_only(df)
          swarm = EthicsSwarm().deliberate(cos)
          assert cos.composite >= 0.70, f'COS {cos.composite:.4f} < 0.70 threshold!'
          assert swarm['approved'], 'EthicsSwarm did not approve!'
          print(f'CATERYA gate passed: COS={cos.composite:.4f} swarm={swarm[\"consensus_score\"]:.4f}')
          "
```
