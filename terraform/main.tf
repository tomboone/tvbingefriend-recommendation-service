# Main Terraform configuration for recommendation service
# This orchestrates both the pipeline and API modules

locals {
  service_name = "tvbf-recommendation-service"
  short_svc_name = "tvbfrecsvc"
}

# Reference shared infrastructure
data "terraform_remote_state" "shared" {
  backend = "azurerm"
  config = {
    resource_group_name  = var.tf_shared_resource_group_name
    storage_account_name = var.tf_shared_storage_account_name
    container_name       = var.tf_shared_container_name
    key                  = var.tf_shared_key
  }
}

data "azurerm_resource_group" "existing" {
  name = data.terraform_remote_state.shared.outputs.resource_group_name
}

data "azurerm_storage_account" "existing" {
  name                     = data.terraform_remote_state.shared.outputs.storage_account_name
  resource_group_name      = data.terraform_remote_state.shared.outputs.resource_group_name
}

data "azurerm_container_registry" "existing" {
  name                = data.terraform_remote_state.shared.outputs.acr_name
  resource_group_name = data.terraform_remote_state.shared.outputs.acr_rg_name
}

data "azurerm_mysql_flexible_server" "existing" {
  name                = data.terraform_remote_state.shared.outputs.mysql_server_name
  resource_group_name = data.terraform_remote_state.shared.outputs.mysql_server_resource_group_name
}

data "azurerm_service_plan" "existing" {
  name                = data.terraform_remote_state.shared.outputs.app_service_plan_name
  resource_group_name = data.terraform_remote_state.shared.outputs.app_service_plan_resource_group
}

resource "azurerm_mysql_flexible_database" "main" {
  name                = local.short_svc_name
  resource_group_name = data.azurerm_mysql_flexible_server.existing.resource_group_name
  server_name         = data.azurerm_mysql_flexible_server.existing.name
  charset             = "utf8mb4"
  collation           = "utf8mb4_unicode_ci"
}

# Generate random passwords for database users
resource "random_password" "db_password" {
  length  = 32
  special = false
}

# Create MySQL user for production
resource "mysql_user" "main" {
  user               = "${local.short_svc_name}_user"
  host               = "%"
  plaintext_password = random_password.db_password.result
}

# Grant permissions to production user
resource "mysql_grant" "main" {
  user       = mysql_user.main.user
  host       = mysql_user.main.host
  database   = azurerm_mysql_flexible_database.main.name
  privileges = ["ALL PRIVILEGES"]
}

# Recommendation API module (Function App)
module "recommendation_api" {
  source = "./modules/recommendation-api"

  # App service plan
  app_service_plan_id = data.azurerm_service_plan.existing.id

  # Log analytics workspace
  log_analytics_workspace_name                = data.terraform_remote_state.shared.outputs.log_analytics_workspace_name
  log_analytics_workspace_resource_group_name = data.terraform_remote_state.shared.outputs.log_analytics_workspace_resource_group_name

  # Azure function
  service_name        = local.service_name
  resource_group_name = data.azurerm_resource_group.existing.name
  location            = data.azurerm_resource_group.existing.location
  allowed_origins    = var.allowed_origins

  # Storage account
  storage_account_name                    = data.azurerm_storage_account.existing.name
  storage_account_access_key              = data.azurerm_storage_account.existing.primary_access_key

  # MySQL flexible server
  mysql_fqdn    = data.azurerm_mysql_flexible_server.existing.fqdn
  mysql_user    = mysql_user.main.user
  mysql_pwd     = random_password.db_password.result
  mysql_db      = azurerm_mysql_flexible_database.main.name
  mysql_charset = azurerm_mysql_flexible_database.main.charset
}

# Recommendation Pipeline module (Container Instance)
module "recommendation_pipeline" {
  source = "./modules/recommendation-pipeline"

  # Shared infrastructure references
  resource_group_name               = data.azurerm_resource_group.existing.name
  location                          = data.azurerm_resource_group.existing.location
  storage_primary_connection_string = data.azurerm_storage_account.existing.primary_connection_string
  storage_container_name            = data.terraform_remote_state.shared.outputs.storage_containers["recommendation-data"]

  # ACR
  acr_login_server   = data.azurerm_container_registry.existing.login_server
  acr_admin_username = data.azurerm_container_registry.existing.admin_username
  acr_admin_password = data.azurerm_container_registry.existing.admin_password

  # Pipeline configuration
  image_name = var.pipeline_image_name
  image_tag  = var.pipeline_image_tag

  cpu_cores     = var.pipeline_cpu_cores
  memory_in_gb  = var.pipeline_memory_in_gb

  # Service URLs
  show_service_url = var.show_service_url

  # MySQL flexible server
  mysql_fqdn    = data.azurerm_mysql_flexible_server.existing.fqdn
  mysql_user    = mysql_user.main.user
  mysql_pwd     = random_password.db_password.result
  mysql_db      = azurerm_mysql_flexible_database.main.name
  mysql_charset = azurerm_mysql_flexible_database.main.charset
}

# Logic App Scheduler module (triggers pipeline weekly)
module "logic_app_scheduler" {
  source = "./modules/logic-app-scheduler"

  logic_app_name      = "${local.service_name}-scheduler"
  resource_group_name = data.azurerm_resource_group.existing.name
  location            = data.azurerm_resource_group.existing.location

  # Container to trigger
  container_group_id   = module.recommendation_pipeline.container_group_id
  container_group_name = module.recommendation_pipeline.container_group_name

  # Schedule configuration (can be overridden in terraform.tfvars)
  schedule_frequency    = var.pipeline_schedule_frequency
  schedule_interval     = var.pipeline_schedule_interval
  schedule_time         = var.pipeline_schedule_time
  schedule_days_of_week = var.pipeline_schedule_days_of_week
}
