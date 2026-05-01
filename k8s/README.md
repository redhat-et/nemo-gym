# NeMo-Gym on OpenShift with KubeRay

Deploy NeMo-Gym benchmarks on OpenShift using RHOAI-managed KubeRay for distributed Ray task execution. The base manifests are benchmark-agnostic — each benchmark gets its own Kustomize overlay (see `overlays/code-gen/` for an example).

## Architecture

```
┌─ Namespace: gym ─────────────────────────────────────────────────┐
│                                                                   │
│  RayCluster: gym-ray                                              │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Head Pod (coordination, dashboard on :8265)                │  │
│  │  Worker Pods (2x, execute @ray.remote tasks)                │  │
│  │  Resources Worker Pod ← runs benchmark server on :9080      │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Deployment: gym-agent (simple_agent on :8080)                    │
│  Deployment: gym-model (vllm_model proxy on :8080)                │
│                                                                   │
│  Services: gym-agent-svc, gym-model-svc, gym-resources-svc        │
│  Route: gym-ray-dashboard (Ray dashboard, edge TLS)               │
└───────────────────────────────────────────────────────────────────┘
        │
        │ HTTPS
        ▼
  External LLM endpoint (OpenAI-compatible /v1/chat/completions)
```

**Data flow:** Client → agent `/run` → model `/v1/responses` → external LLM → agent extracts code → resources `/verify` → Ray workers execute unit tests → reward (0.0 or 1.0)

## Directory Structure

```
k8s/
├── Dockerfile                        # Multi-stage: resources, agent, model targets
├── entrypoint.sh                     # OpenShift arbitrary UID handler
├── README.md
├── base/                             # Generic manifests (no benchmark-specific config)
│   ├── kustomization.yaml
│   ├── namespace.yaml                # Namespace: gym
│   ├── serviceaccount.yaml           # ServiceAccount: gym-sa
│   ├── configmap.yaml                # Base Hydra config (agent + model only)
│   ├── raycluster.yaml               # Ray head, workers, resources worker group
│   ├── deployment-agent.yaml         # simple_agent (skips Ray)
│   ├── deployment-model.yaml         # vllm_model proxy (skips Ray)
│   ├── service-*.yaml                # ClusterIP services
│   ├── networkpolicy.yaml            # Internal + ingress + monitoring + Ray workers
│   └── route-dashboard.yaml          # Ray dashboard (edge TLS)
└── overlays/
    └── code-gen/                     # code_gen benchmark overlay (use as template)
        ├── kustomization.yaml        # Patches ConfigMap + RayCluster for code_gen
        ├── configmap-patch.yaml      # code_gen resources server Hydra config
        └── secret.yaml               # LLM credentials (gitignored, template committed)
```

## Prerequisites

### OpenShift cluster requirements

| Component | Requirement | How to verify |
|-----------|-------------|---------------|
| OpenShift | 4.14+ | `oc version` |
| RHOAI (Red Hat OpenShift AI) | 2.x+ with KubeRay operator enabled | `oc get csv -n redhat-ods-applications \| grep rhods` |
| KubeRay operator | Managed by RHOAI — must be running | `oc get deployment -n redhat-ods-applications -l app.kubernetes.io/name=kuberay` |
| Worker nodes | At least 1 CPU worker with 32Gi+ RAM (for Ray head + workers + resources server) | `oc get nodes -l node-role.kubernetes.io/worker` |
| Default StorageClass | Any (only emptyDir volumes used, but cluster must be functional) | `oc get sc` |
| Namespace creation | User must be able to create projects | `oc auth can-i create projects` |
| Image pull access | Cluster nodes must reach `quay.io` (or use a pull secret for private registries) | `oc debug node/<node> -- curl -s https://quay.io/v2/` |

RHOAI's KubeRay operator automatically injects mTLS certificates and OAuth proxies into RayCluster pods. The deployment manifests are designed to work with this — no manual TLS configuration is needed.

### Workstation requirements

- `oc` CLI authenticated to the cluster (`oc login`)
- `podman` (or `docker`) for building linux/amd64 images
- An OpenAI-compatible LLM endpoint accessible from the cluster (vLLM, RHOAI Model-as-a-Service, OpenAI, etc.) — the cluster pods must be able to reach this endpoint over HTTPS

## Deployment Defaults

The manifests deploy into a namespace called **`gym`** by default. All resource names are prefixed with `gym-` (e.g., `gym-ray`, `gym-agent-svc`). These defaults work out of the box — you only need to change them if you want to run multiple instances or follow a naming convention.

### Changing the namespace

Edit one line in `k8s/base/kustomization.yaml`:

```yaml
namespace: my-namespace    # change from "gym" to whatever you want
```

Kustomize applies this to all resources at render time. Service discovery uses short names (e.g., `gym-agent-svc`) that resolve within whichever namespace you deploy to — no FQDN edits needed.

If the namespace doesn't exist yet, either create it first (`oc new-project my-namespace`) or let the included `namespace.yaml` create it. Note that `namespace.yaml` creates a namespace called `gym` — if you change the Kustomize namespace, also update `namespace.yaml` or remove it and create the namespace yourself.

### Customizing resource names

The `gym-` prefix on all resource names (services, deployments, RayCluster) is hardcoded in the YAML files. If you need different names, edit the base manifests directly. Be sure to update all cross-references (Service selectors, env vars, ConfigMap refs, etc.).

## Quick Start

### 1. Build and push images

From the Gym repo root:

```bash
podman build --platform linux/amd64 -f k8s/Dockerfile --target resources -t quay.io/redhat-et/nemo-gym-resources:latest .
podman build --platform linux/amd64 -f k8s/Dockerfile --target agent -t quay.io/redhat-et/nemo-gym-agent:latest .
podman build --platform linux/amd64 -f k8s/Dockerfile --target model -t quay.io/redhat-et/nemo-gym-model:latest .

podman push quay.io/redhat-et/nemo-gym-resources:latest
podman push quay.io/redhat-et/nemo-gym-agent:latest
podman push quay.io/redhat-et/nemo-gym-model:latest
```

Ensure the image repos are set to **public** on quay.io, or add a pull secret to the `gym` namespace.

### 2. Configure the LLM endpoint

Before deploying, create the secret file with your LLM credentials. The overlay expects `k8s/overlays/code-gen/secret.yaml` (gitignored — it won't exist on a fresh clone):

```bash
cat > k8s/overlays/code-gen/secret.yaml <<'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: gym-secrets
  namespace: gym
  labels:
    app.kubernetes.io/part-of: nemo-gym
type: Opaque
stringData:
  POLICY_BASE_URL: "https://your-llm-endpoint/v1"
  POLICY_API_KEY: "your-api-key"
  POLICY_MODEL_NAME: "your-model-name"
EOF
```

| Field | What to set | Example |
|-------|-------------|---------|
| `POLICY_BASE_URL` | OpenAI-compatible API base URL (must end with `/v1`) | `https://my-vllm.example.com/v1` |
| `POLICY_API_KEY` | API key or bearer token. Use any value if the endpoint has no auth. | `sk-abc123` |
| `POLICY_MODEL_NAME` | Model identifier as the endpoint expects it in the `model` field | `meta-llama/Llama-3.1-8B-Instruct` |

### 3. Deploy

```bash
oc apply -k k8s/overlays/code-gen/
```

Pods will pull images, start Ray, and connect to your LLM endpoint. If you need to update credentials later:

```bash
# Edit k8s/overlays/code-gen/secret.yaml, then:
oc apply -k k8s/overlays/code-gen/
oc rollout restart deployment/gym-agent deployment/gym-model -n gym
```

### 4. Verify

Wait for all pods to reach Running/Ready:

```bash
oc get pods -n gym -w
```

Expected state:

| Pod | Ready | Description |
|-----|-------|-------------|
| `gym-ray-head-*` | 2/2 | Ray head + OAuth proxy |
| `gym-ray-gym-workers-worker-*` (x2) | 1/1 | Ray code execution workers |
| `gym-ray-gym-resources-worker-*` | 1/1 | Resources server (code_gen) + Ray node |
| `gym-agent-*` | 1/1 | Agent server (simple_agent) |
| `gym-model-*` | 1/1 | Model proxy (vllm_model) |

Check the resources server connected to Ray:

```bash
oc exec $(oc get pod -n gym -l ray.io/group=gym-resources -o name) -n gym -- tail -5 /tmp/nemo-gym-server.log
```

You should see `Connected to Ray cluster` and `Uvicorn running on http://0.0.0.0:9080`.

### 5. Smoke test

```bash
# Port-forward the agent
oc port-forward -n gym svc/gym-agent-svc 8080:8080 &

# Send a test request
curl -s -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d @<(head -1 resources_servers/code_gen/data/example.jsonl)

# Clean up
kill %1
```

A successful response contains a `reward` field (0.0 or 1.0) and the model's generated code.

### 6. Collect rollouts

For benchmarking, use `ng_collect_rollouts` targeting the deployed agent via port-forward:

```bash
oc port-forward -n gym svc/gym-agent-svc 8080:8080 &

ng_collect_rollouts \
  +agent_name=simple_agent_instance \
  +input_jsonl_fpath=resources_servers/code_gen/data/example.jsonl \
  +output_jsonl_fpath=results/rollouts.jsonl \
  +num_repeats=5 \
  "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}" \
  "+agent_server_url=http://localhost:8080"
```

Profile results:

```bash
ng_reward_profile \
  +input_jsonl_fpath=resources_servers/code_gen/data/example.jsonl \
  +rollouts_jsonl_fpath=results/rollouts.jsonl \
  +output_jsonl_fpath=results/profiled.jsonl \
  +pass_threshold=1.0
```

## Customization

### Using a different LLM

Update the Secret and restart the model deployment:

```bash
oc set data secret/gym-secrets -n gym \
  --from-literal=POLICY_BASE_URL="https://new-endpoint/v1" \
  --from-literal=POLICY_API_KEY="new-key" \
  --from-literal=POLICY_MODEL_NAME="new-model"

oc rollout restart deployment/gym-model -n gym
```

### Adding a new benchmark

The base manifests are benchmark-agnostic. Each benchmark gets its own Kustomize overlay with three files. Use `overlays/code-gen/` as a starting point:

```bash
cp -r k8s/overlays/code-gen k8s/overlays/my-benchmark
```

#### File 1: `secret.yaml` — LLM credentials

Provides the connection details for the OpenAI-compatible LLM that the model proxy calls.

```yaml
stringData:
  POLICY_BASE_URL: "https://your-llm-endpoint/v1"   # Must end with /v1
  POLICY_API_KEY: "your-api-key"                     # API key or token
  POLICY_MODEL_NAME: "your-model-name"               # Model ID as the endpoint expects it
```

| Field | What it is | Example |
|-------|-----------|---------|
| `POLICY_BASE_URL` | Base URL of an OpenAI-compatible API. Must include `/v1`. The model proxy appends `/chat/completions`. | `https://my-vllm.example.com/v1` |
| `POLICY_API_KEY` | Bearer token for the LLM API. Set to any value (e.g., `unused`) if the endpoint doesn't require auth. | `sk-abc123...` |
| `POLICY_MODEL_NAME` | Model identifier passed in the `model` field of API requests. Must match what the serving endpoint expects. | `meta-llama/Llama-3.1-8B-Instruct` |

This file is gitignored. The checked-in template has placeholder values — replace them before deploying.

#### File 2: `configmap-patch.yaml` — Benchmark server config

Defines the NeMo-Gym Hydra config for your resources server. This replaces the base ConfigMap entirely.

The key section to change is `resources_instance` — everything else (agent, model, Ray settings) stays the same:

```yaml
    resources_instance:                    # Keep this name — the agent references it
      resources_servers:
        my_server:                         # ← Your server's directory name under resources_servers/
          entrypoint: app.py               # Usually app.py
          host: "${oc.env:RESOURCES_HOST}" # Don't change — resolved at runtime
          port: 9080                       # Don't change — must avoid Ray metrics on 8080
          domain: coding                   # Domain hint (coding, math, etc.)
          # Add any server-specific config fields below:
          num_processes: 4
          timeout_secs: 30
```

The fields under your server name are passed directly to your server's Pydantic config model. Check your server's `app.py` for available config fields.

#### File 3: `kustomization.yaml` — Wiring it together

Two JSON patches tell the RayCluster which server to start and which config section to load:

```yaml
patches:
  - path: configmap-patch.yaml
    target:
      kind: ConfigMap
      name: gym-config
  - patch: |-
      - op: replace
        path: /spec/workerGroupSpecs/1/template/spec/containers/0/env/2/value
        value: "python resources_servers/my_server/app.py"
      - op: replace
        path: /spec/workerGroupSpecs/1/template/spec/containers/0/env/4/value
        value: "resources_instance"
    target:
      kind: RayCluster
      name: gym-ray
```

| Patch | What it sets | Change to |
|-------|-------------|-----------|
| `env/2/value` | `NEMO_GYM_SERVER_ENTRYPOINT` — the Python command to start your server | `python resources_servers/<your_server>/app.py` |
| `env/4/value` | `NEMO_GYM_CONFIG_PATH` — which top-level config key your server reads | `resources_instance` (keep as-is unless you renamed it) |

#### Deploy and test

```bash
# Deploy
oc apply -k k8s/overlays/my-benchmark/

# Apply your real secret (the template has placeholders)
oc create secret generic gym-secrets -n gym \
  --from-literal=POLICY_BASE_URL="https://actual-endpoint/v1" \
  --from-literal=POLICY_API_KEY="actual-key" \
  --from-literal=POLICY_MODEL_NAME="actual-model" \
  --dry-run=client -o yaml | oc apply -f -

# Restart pods to pick up the secret
oc rollout restart deployment/gym-agent deployment/gym-model -n gym

# Wait for all pods
oc get pods -n gym -w

# Smoke test (use your benchmark's example data)
oc port-forward -n gym svc/gym-agent-svc 8080:8080 &
curl -s -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d @<(head -1 resources_servers/my_server/data/example.jsonl)
kill %1
```

### Scaling Ray workers

Edit `k8s/base/raycluster.yaml` and adjust the `gym-workers` group:

```yaml
workerGroupSpecs:
  - groupName: gym-workers
    replicas: 4        # increase for more parallelism
    minReplicas: 2
    maxReplicas: 8
```

Then reapply: `oc apply -k k8s/overlays/code-gen/`

### Adjusting memory

- **Agent/model pods**: 2Gi each (no Ray runtime). Adjust in `deployment-agent.yaml` / `deployment-model.yaml`.
- **Resources worker**: 8Gi (includes Ray node). Adjust in `raycluster.yaml` under `gym-resources` group.
- **Ray execution workers**: 8Gi each. Adjust in `raycluster.yaml` under `gym-workers` group.

## How It Works

### RHOAI mTLS integration

RHOAI's KubeRay operator injects mTLS certificates into all RayCluster pods. The resources server runs inside a RayCluster worker group pod so it receives valid TLS certs automatically. A `postStart` lifecycle hook starts the NeMo-Gym server after Ray joins the cluster. The agent and model servers run as standard Deployments and skip Ray initialization entirely (`NEMO_GYM_SKIP_RAY_INIT=1`).

### Port layout

| Server | Port | Notes |
|--------|------|-------|
| Agent (simple_agent) | 8080 | Standard HTTP |
| Model (vllm_model) | 8080 | Standard HTTP |
| Resources (benchmark) | 9080 | Non-default to avoid conflict with Ray metrics on 8080 |
| Ray GCS | 6379 | Internal, mTLS |
| Ray Dashboard | 8265 | Exposed via Route |

### NetworkPolicies

RHOAI creates NetworkPolicies restricting Ray cluster traffic. An additional `allow-gym-to-ray-workers` policy permits the agent and model pods to reach the resources server on the Ray worker group.

## Teardown

```bash
oc delete -k k8s/overlays/code-gen/
oc delete project gym
```

## Troubleshooting

**General debugging — check recent events:**
```bash
oc get events -n gym --sort-by='.lastTimestamp' | tail -20
```

**Resources pod not reaching Ready:**
```bash
oc exec $(oc get pod -n gym -l ray.io/group=gym-resources -o name) -n gym -- cat /tmp/nemo-gym-server.log
```

**Agent returns 405 or connection errors:**
Check NetworkPolicies allow traffic from agent to resources pods:
```bash
oc get networkpolicy -n gym
```

**Model returns connection errors:**
Verify the LLM endpoint is reachable from within the cluster:
```bash
oc exec $(oc get pod -n gym -l app.kubernetes.io/name=gym-model -o name) -n gym -- \
  curl -s "$POLICY_BASE_URL/models" -H "Authorization: Bearer $POLICY_API_KEY"
```

**Ray dashboard:**
```bash
oc get route gym-ray-dashboard -n gym -o jsonpath='{.spec.host}'
```
