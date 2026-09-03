# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Unity Catalog setup
# MAGIC
# MAGIC Makes sure the catalog, the bronze/silver/gold schemas and a checkpoint volume exist, and that
# MAGIC the raw CSVs are visible at `raw_path`.
# MAGIC
# MAGIC Two ways to land the raw files, both supported here:
# MAGIC
# MAGIC 1. **Uploaded through the workspace UI** into a Unity Catalog volume (Catalog > Volumes > Upload).
# MAGIC    Set `raw_path` to that volume path, leave `storage_account` / `access_connector_id` empty.
# MAGIC 2. **Your own ADLS Gen2 container** (via `infra/azure/provision.sh` + `upload_raw.sh`). Set
# MAGIC    `storage_account` and `access_connector_id`; this notebook then creates the storage
# MAGIC    credential, external location and an external volume over `raw/`.
# MAGIC
# MAGIC **Concepts (exam topics)**
# MAGIC - **Catalog / schema / table / volume** — the three-level namespace. A *volume* exposes files
# MAGIC   (not tables) at `/Volumes/<catalog>/<schema>/<volume>/...` so Auto Loader can read CSVs.
# MAGIC - **Managed vs external** — managed tables/volumes live under the catalog or metastore storage
# MAGIC   root; external ones point at a path you own. Dropping a managed object deletes its files,
# MAGIC   dropping an external one does not.
# MAGIC - **Storage credential** — wraps an Azure *access connector* (managed identity) that has
# MAGIC   *Storage Blob Data Contributor* on the account. No keys or SAS tokens in notebooks.
# MAGIC - **External location** — storage credential + URL. UC grants (`READ FILES`, `CREATE TABLE`,
# MAGIC   ...) are checked against it before any path under that URL is touched.

# COMMAND ----------

dbutils.widgets.text("catalog", "learningdatabricks")
dbutils.widgets.text("raw_path", "/Volumes/learningdatabricks/home-credit-default-risk/home-credit-default-risk")
dbutils.widgets.text("storage_account", "")
dbutils.widgets.text("container", "home-credit")
dbutils.widgets.text("access_connector_id", "")

catalog = dbutils.widgets.get("catalog")
raw_path = dbutils.widgets.get("raw_path").rstrip("/")
storage_account = dbutils.widgets.get("storage_account")
container = dbutils.widgets.get("container")
access_connector_id = dbutils.widgets.get("access_connector_id")

use_own_storage = bool(storage_account and access_connector_id)
print(f"catalog={catalog}\nraw_path={raw_path}\nown ADLS storage={use_own_storage}")

# COMMAND ----------

# MAGIC %md ## Option 2 only: storage credential + external location over your ADLS container

# COMMAND ----------

if use_own_storage:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import AzureManagedIdentity

    root_url = f"abfss://{container}@{storage_account}.dfs.core.windows.net"
    credential_name = f"{catalog}_adls_cred"
    location_name = f"{catalog}_adls"

    # No SQL DDL exists for storage credentials, so use the SDK (preinstalled on DBR 13.3+)
    w = WorkspaceClient()
    if credential_name in {c.name for c in w.storage_credentials.list()}:
        print(f"storage credential {credential_name} already exists")
    else:
        w.storage_credentials.create(
            name=credential_name,
            azure_managed_identity=AzureManagedIdentity(access_connector_id=access_connector_id),
            comment="Access connector for the Home Credit ADLS Gen2 account",
        )
        print(f"created storage credential {credential_name}")

    spark.sql(f"""
    CREATE EXTERNAL LOCATION IF NOT EXISTS {location_name}
      URL '{root_url}/'
      WITH (STORAGE CREDENTIAL {credential_name})
      COMMENT 'Root of the home-credit container'
    """)
    # Managed tables/volumes for this catalog land under <container>/managed/
    spark.sql(f"""
    CREATE CATALOG IF NOT EXISTS {catalog}
      MANAGED LOCATION '{root_url}/managed'
      COMMENT 'Home Credit default risk - medallion pipeline'
    """)
else:
    # Don't even issue CREATE CATALOG IF NOT EXISTS when the catalog exists: on accounts with
    # Default Storage, UC validates the metastore storage root first and raises INVALID_STATE.
    existing_catalogs = {r.catalog for r in spark.sql("SHOW CATALOGS").collect()}
    if catalog in existing_catalogs:
        print(f"catalog {catalog} exists")
    else:
        try:
            spark.sql(f"CREATE CATALOG {catalog} "
                      "COMMENT 'Home Credit default risk - medallion pipeline'")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Catalog {catalog} does not exist and cannot be created without a storage "
                "location. Create it in the UI (Catalog > Create catalog, Default Storage) or "
                "pass storage_account/access_connector_id to use your own ADLS container."
            ) from e

# COMMAND ----------

# MAGIC %md ## Schemas and volumes

# COMMAND ----------

for schema in ("bronze", "silver", "gold"):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

if use_own_storage:
    spark.sql(f"""
    CREATE EXTERNAL VOLUME IF NOT EXISTS {catalog}.bronze.raw
      LOCATION '{root_url}/raw'
      COMMENT 'Raw Kaggle CSVs, uploaded by infra/azure/upload_raw.sh'
    """)

# Managed volume for Auto Loader checkpoints / schema tracking
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.bronze.checkpoints")

# COMMAND ----------

# MAGIC %md ## Verify: the raw CSVs must be visible at raw_path

# COMMAND ----------

files = dbutils.fs.ls(raw_path)
display(files)
names = {f.name for f in files}
expected = {
    "application_train.csv", "application_test.csv", "bureau.csv", "bureau_balance.csv",
    "previous_application.csv", "POS_CASH_balance.csv", "installments_payments.csv",
    "credit_card_balance.csv",
}
missing = expected - names
assert not missing, f"missing in {raw_path}: {sorted(missing)}"

# COMMAND ----------

display(spark.sql(f"SHOW SCHEMAS IN {catalog}"))
display(spark.sql(f"SHOW VOLUMES IN {catalog}.bronze"))
