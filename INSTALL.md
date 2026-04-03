# INSTALL.md — Eco AI Data Center v2.0
### Panduan Instalasi Lengkap

**CateryaTech** | Ary HH | cateryatech@proton.me

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.12 recommended |
| pip | 23+ | `pip install --upgrade pip` |
| OS | Linux / macOS / Windows WSL2 | |
| RAM | 4GB minimum | 8GB untuk quantum features |
| Storage | 2GB | Untuk dependencies |

---

## 1. Clone & Virtual Environment

```bash
git clone https://github.com/cateryatech/Eco-AI-Data-Center
cd Eco-AI-Data-Center

# Create venv
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

---

## 2. Install Dependencies

### Minimal (core features only)

```bash
pip install -r requirements.txt
```

### Full Enterprise Stack

```bash
# Core + all optional integrations
pip install -r requirements.txt

# Blockchain (Polygon + IPFS)
pip install web3==6.20.1 py-solc-x

# Advanced encryption
pip install cryptography==42.0.8

# Quantum computing (optional, falls back to classical)
pip install pennylane==0.39.0 pennylane-lightning

# Production task queue
pip install celery==5.4.0 redis==5.0.8

# Time-series database
pip install influxdb-client==3.10.0

# Stripe billing
pip install stripe==10.12.0

# Analytics forwarding (optional)
pip install mixpanel requests

# ML/monitoring
pip install torch scikit-learn

# API server
pip install fastapi uvicorn strawberry-graphql

# Testing
pip install pytest pytest-asyncio httpx
```

---

## 3. Environment Variables

Copy template dan isi dengan credentials Anda:

```bash
cp .env.example .env
nano .env
```

### .env.example

```bash
# ─── Application ──────────────────────────────────
APP_ENV=development               # development | staging | production
APP_SECRET_KEY=change-this-to-a-random-64-char-string
LOG_LEVEL=INFO

# ─── Encryption ───────────────────────────────────
MASTER_ENCRYPTION_KEY=your-enterprise-master-key-min-32-chars
# Generate: python3 -c "import secrets; print(secrets.token_hex(32))"

# ─── JWT Auth ─────────────────────────────────────
JWT_SECRET_KEY=your-jwt-secret-key-change-in-production
JWT_ACCESS_EXPIRE_MINUTES=60
JWT_REFRESH_EXPIRE_DAYS=7

# ─── Database (optional — defaults to in-memory) ──
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-influxdb-token
INFLUXDB_ORG=cateryatech
INFLUXDB_BUCKET=eco_ai_sensors

# ─── Redis (for Celery background tasks) ──────────
REDIS_URL=redis://localhost:6379/0

# ─── Blockchain (optional — falls back to local) ──
POLYGON_RPC_URL=https://polygon-rpc.com
POLYGON_PRIVATE_KEY=0xYourPrivateKey
POLYGON_NETWORK=polygon              # polygon | mumbai (testnet)
PINATA_API_KEY=your-pinata-api-key
PINATA_SECRET=your-pinata-secret

# ─── Stripe Billing ───────────────────────────────
STRIPE_SECRET_KEY=sk_test_...       # Use sk_live_... in production
STRIPE_WEBHOOK_SECRET=whsec_...     # From Stripe dashboard
STRIPE_PRICE_STARTER=price_xxx
STRIPE_PRICE_GROWTH=price_yyy
STRIPE_PRICE_ENTERPRISE=price_zzz
STRIPE_PRICE_GOVERNMENT=price_www

# ─── Usage Analytics (optional) ───────────────────
MIXPANEL_TOKEN=your-mixpanel-token
GA_MEASUREMENT_ID=G-XXXXXXXXXX
GA_API_SECRET=your-ga4-api-secret

# ─── Admin Credentials (CHANGE BEFORE PRODUCTION) ─
ADMIN_PASSWORD=Admin@EcoAI2025!
```

---

## 4. Launch

### Development

```bash
# Streamlit dashboard
streamlit run app.py
# → http://localhost:8501

# FastAPI + GraphQL (separate terminal)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs
# → http://localhost:8000/graphql
```

### Background Workers (optional)

```bash
# Redis must be running first
redis-server &

# Start Celery worker
celery -A workers.tasks worker --loglevel=info --concurrency=4

# Start Celery beat (scheduled tasks)
celery -A workers.tasks beat --loglevel=info
```

---

## 5. Verify Installation

```bash
# Quick smoke test
python3 -c "
import sys; sys.path.insert(0, '.')
from caterya_framework.evaluator import CATERYAEvaluator
from monetization.analytics import UsageTracker, UsageEventType
from monetization.billing import BillingService
from reports.insights import InsightsEngine
from blockchain_integration import BlockchainIntegration
from security.auth import AuthService

print('✅ CATERYA Framework OK')
print('✅ Monetization OK')
print('✅ Insights Engine OK')
print('✅ Blockchain Integration OK')
print('✅ Security Auth OK')
print()
print('All modules imported successfully — ready for deployment!')
"

# Full test suite
python3 tests/test_monetization_e2e.py
python3 tests/test_blockchain_agentic.py
```

---

## 6. Troubleshooting

### `ImportError: No module named 'cryptography'`
```bash
pip install cryptography==42.0.8
```

### `ImportError: No module named 'pennylane'`
```bash
# Falls back to classical simulation automatically — no action needed
# Or install: pip install pennylane
```

### `web3.exceptions.ProviderConnectionError`
```bash
# Blockchain uses local registry fallback automatically
# Set POLYGON_RPC_URL in .env if you want real on-chain transactions
```

### Streamlit port conflict
```bash
streamlit run app.py --server.port 8502
```

### Windows: encoding error
```bash
# Use WSL2 or add to .env:
PYTHONIOENCODING=utf-8
```

---

## 7. Docker (Optional)

```dockerfile
# Dockerfile (minimal)
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

```bash
docker build -t eco-ai-datacenter .
docker run -p 8501:8501 -p 8000:8000 --env-file .env eco-ai-datacenter
```

---

## 8. Production Checklist

- [ ] Ganti semua password default di `.env`
- [ ] Set `APP_ENV=production`
- [ ] Enable HTTPS (Nginx reverse proxy atau Cloudflare)
- [ ] Set `JWT_SECRET_KEY` ke random 64-char string
- [ ] Set `MASTER_ENCRYPTION_KEY` ke nilai unik dan simpan di secrets manager
- [ ] Konfigurasi Stripe dengan `sk_live_...` key
- [ ] Set `POLYGON_PRIVATE_KEY` untuk blockchain features
- [ ] Deploy Redis untuk Celery workers
- [ ] Setup InfluxDB untuk sensor data persistence
- [ ] Review `SECURITY.md` untuk threat model lengkap

---

*Lihat [DEPLOYMENT.md](DEPLOYMENT.md) untuk panduan AWS, Heroku, dan Kubernetes.*
