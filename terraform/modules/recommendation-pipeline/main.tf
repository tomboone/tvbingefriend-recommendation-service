# Recommendation Pipeline Module - Azure Container Instance

# Container Instance for recommendation pipeline
resource "azurerm_container_group" "pipeline" {
  name                = "recommendation-pipeline"
  location            = var.location
  resource_group_name = var.resource_group_name
  ip_address_type     = "None"
  os_type             = "Linux"
  restart_policy      = "OnFailure"

  image_registry_credential {
    server   = var.acr_login_server
    username = var.acr_admin_username
    password = var.acr_admin_password
  }

  container {
    name   = "recommendation-pipeline"
    image  = "${var.acr_login_server}/${var.image_name}:${var.image_tag}"
    cpu    = var.cpu_cores
    memory = var.memory_in_gb

    # Non-sensitive environment variables
    environment_variables = {
      SHOW_SERVICE_URL       = var.show_service_url
      STORAGE_CONTAINER_NAME = var.storage_container_name
      USE_BLOB_STORAGE       = "true"
    }

    # Sensitive environment variables
    secure_environment_variables = {
      AZURE_STORAGE_CONNECTION_STRING = var.storage_primary_connection_string
      DATABASE_URL                    = "mysql+pymysql://${var.mysql_user}:${var.mysql_pwd}@${var.mysql_fqdn}:3306/${var.mysql_db}?charset=${var.mysql_charset}"
    }
  }
}
