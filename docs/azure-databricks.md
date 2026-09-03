# Azure Databricks track

Hands-on exam prep: the same Home Credit pipeline that runs locally, rebuilt as a
Delta Lake medallion pipeline on Azure Databricks with Unity Catalog, Auto Loader,
MLflow and a Databricks Job. The local pandas/LightGBM pipeline keeps working as before.

```
data/raw/*.csv ──upload_raw.sh──▶ ADLS Gen2  home-credit/raw/
                                       │  external volume /Volumes/home_credit/bronze/raw
                                       ▼
   00 setup UC ─▶ 01 bronze (Auto Loader) ─▶ 02 silver (MERGE, constraints, OPTIMIZE)
                                       ─▶ 03 gold features (PySpark port, PK = feature table)
                                       ─▶ 04 train LightGBM (MLflow, UC model registry, @champion)
                                       ─▶ 05 batch score (pyfunc spark_udf ─▶ gold.predictions)
```

## Layout

| Path | What |
|---|---|
| `infra/azure/provision.sh` | Resource group, ADLS Gen2 account + container, Premium Databricks workspace, access connector, RBAC. Idempotent. Writes `infra/azure/.env.azure`. |
| `infra/azure/upload_raw.sh` | Uploads `data/raw/*.csv` to `raw/` in the container with your AD login. |
| `databricks/databricks.yml` | Databricks Asset Bundle (DAB). Variables come from `.env.azure`. |
| `databricks/resources/medallion_job.yml` | One Job, six notebook tasks, on serverless compute (classic job-cluster variant kept as a comment). |
| `databricks/notebooks/0*.py` | The pipeline, as Databricks source-format notebooks with concept notes at the top of each. |
| `src/home_credit/data/loader.py` | Local pipeline can now read the CSVs straight from ADLS via `AZURE_DATA_URI`. |

## Runbook

### Fast path: workspace already exists and the CSVs are already in a volume

Skip steps 1 and 2. Authenticate the CLI (step 3), then `make dbx-deploy`. The bundle defaults
already point at the existing workspace catalog and volume:

| Variable | Default |
|---|---|
| `catalog` | `learningdatabricks` |
| `raw_path` | `/Volumes/learningdatabricks/home-credit-default-risk/home-credit-default-risk` |

Override with `make dbx-deploy DBX_VARS='--var="raw_path=/Volumes/<catalog>/<schema>/<volume>"'`.

Notebook 00 then only creates the bronze/silver/gold schemas and a checkpoint volume, and
checks the eight CSVs are visible at `raw_path`. The storage-credential / external-location
branch is skipped because `storage_account` and `access_connector_id` are empty.

### 0. One-time local setup

```bash
az logout && az login --tenant f03ee0cd-1574-4b8a-bd93-cf21f09a3b20
```

The Databricks CLI (v1.14.1) is installed via winget. Check with `databricks --version`.

### 1. Provision Azure

```bash
make azure-provision
```

Defaults: `eastus`, resource group `rg-home-credit`, storage `sthomecredit<sub-id-prefix>`,
workspace `dbw-home-credit`. Override with env vars, e.g. `AZ_LOCATION=centralindia make azure-provision`.
The workspace takes 5 to 10 minutes. Outputs land in `infra/azure/.env.azure` (gitignored).

### 2. Upload the raw CSVs (2.5 GB)

```bash
make azure-upload
```

### 3. Authenticate the Databricks CLI to the workspace

```bash
databricks auth login --host https://adb-<workspace-id>.<n>.azuredatabricks.net --profile home-credit
```

The URL is in the browser address bar when the workspace is open (also in `.env.azure` if you
used the provision script).

This opens a browser for OAuth (user-to-machine) and stores the profile in `~/.databrickscfg`.
Then, in the same shell:

```bash
export DATABRICKS_CONFIG_PROFILE=home-credit
```

### 4. Deploy and run the bundle

If you brought your own ADLS container, pass its details once:
`DBX_VARS='--var="storage_account=<acct>" --var="access_connector_id=<id>"'`.

```bash
make dbx-validate
```

```bash
make dbx-deploy
```

```bash
make dbx-run
```

`dbx-run` starts the job and streams task status. The job runs on serverless compute because
the subscription's eastus region had `Standard_DS3_v2` stocked out and zero quota for the Dv5
family; serverless sidesteps both. Re-runs are faster because Auto Loader skips files it has already seen.

You can also run individual notebooks interactively from the workspace UI: they are synced to
`/Users/<you>/.bundle/home_credit_default_risk/dev/files/notebooks/`. Attach any UC-enabled
cluster and fill in the widgets at the top.

### 5. Read ADLS from the local pipeline (optional)

Add `AZURE_DATA_URI=abfss://home-credit@<account>.dfs.core.windows.net/raw` to `.env`
(the value is in `.env.azure`). `python scripts/train.py --no-cache` then reads the CSVs
from ADLS using your `az login` session.

## Permissions you need

- Azure: Owner or Contributor + User Access Administrator on the subscription, so the script can
  create role assignments.
- Databricks: the workspace creator is a workspace admin. Creating a **storage credential** and
  **external location** (notebook 00) additionally needs metastore-level privileges. On a fresh
  tenant the account admin has them. If notebook 00 fails with a permission error, open
  https://accounts.azuredatabricks.net, go to Catalog, and either make yourself metastore admin
  or grant `CREATE STORAGE CREDENTIAL` and `CREATE EXTERNAL LOCATION` to your user.
- If Unity Catalog is not enabled for the workspace (older regions), attach it to a metastore
  from the account console first.

## Cost notes

- Premium workspace: no charge while idle. Serverless jobs bill per DBU-second while tasks run;
  budget a few USD per full run. Classic compute needs VM quota in the region (see the comment in
  `medallion_job.yml`).
- Storage: 2.5 GB raw plus Delta copies, cents per month.
- Tear everything down with `az group delete -n rg-home-credit --yes` when finished.

## Exam topic map

| Topic | Where |
|---|---|
| Unity Catalog objects, storage credential, external location, volumes, managed vs external | `00_setup_unity_catalog.py` |
| Auto Loader, schema inference and evolution, checkpoints, `availableNow` trigger, `_metadata` | `01_bronze_autoloader.py` |
| `MERGE INTO`, constraints, `OPTIMIZE` / `ZORDER`, time travel, `overwriteSchema` | `02_silver_clean.py` |
| DataFrame API aggregations, null semantics, feature tables via primary key | `03_gold_features.py` |
| MLflow experiments, UC model registry, signatures, aliases, custom pyfunc | `04_train_lgbm.py` |
| `spark_udf` batch inference, alias-based model resolution | `05_batch_score.py` |
| Jobs / Workflows, task dependencies, job parameters, job clusters, asset bundles | `databricks/resources/medallion_job.yml`, `databricks.yml` |

## Suggested follow-up exercises

1. Convert 01 to 03 into a **Delta Live Tables / Lakeflow Declarative** pipeline with
   `@dlt.table` and expectations (`@dlt.expect_or_drop`).
2. Switch the job to a **classic job cluster** (uncomment the block in `medallion_job.yml`,
   after requesting VM quota in the Azure portal) and compare startup time and cost with serverless.
3. Add a **Databricks SQL** dashboard over `gold.predictions` and an alert on mean score drift.
4. Package `src/home_credit` as a wheel through the bundle `artifacts:` block and import the
   pandas feature code in a notebook instead of the inline PySpark port.
5. Deploy the champion model to **Model Serving** and call it from `home_credit/serving/app.py`.
