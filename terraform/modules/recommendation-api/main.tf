data "azurerm_log_analytics_workspace" "existing" {
  name                = var.log_analytics_workspace_name
  resource_group_name = var.log_analytics_workspace_resource_group_name
}

resource "azurerm_application_insights" "main" {
  name                = var.service_name
  resource_group_name = var.resource_group_name
  location            = var.location
  workspace_id        = data.azurerm_log_analytics_workspace.existing.id
  application_type    = "web"
}

resource "azurerm_linux_function_app" "main" {
  name                       = var.service_name
  resource_group_name        = var.resource_group_name
  location                   = var.location
  service_plan_id            = var.app_service_plan_id
  storage_account_name       = var.storage_account_name
  storage_account_access_key = var.storage_account_access_key

  site_config {
    always_on = true
    application_insights_connection_string = azurerm_application_insights.main.connection_string
    application_insights_key               = azurerm_application_insights.main.instrumentation_key
    application_stack {
      python_version = "3.12"
    }
    cors {
      allowed_origins = var.allowed_origins
      support_credentials = false
    }
  }

  app_settings = {
    DATABASE_URL = "mysql+pymysql://${var.mysql_user}:${var.mysql_pwd}@${var.mysql_fqdn}:3306/${var.mysql_db}?charset=${var.mysql_charset}"
  }
}
