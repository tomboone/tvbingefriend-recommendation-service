# Outputs for recommendation API module

output "function_app_name" {
  description = "Name of the function app"
  value       = azurerm_linux_function_app.main.name
}

output "function_app_url" {
  description = "URL of the function app"
  value       = "https://${azurerm_linux_function_app.main.default_hostname}"
}

output "function_app_id" {
  description = "Resource ID of the function app"
  value       = azurerm_linux_function_app.main.id
}

output "function_app_resource_group_name" {
  description = "Resource group name of the function app"
  value = azurerm_linux_function_app.main.resource_group_name
}

output "function_app_python_version"  {
  description = "Python version of the function app"
  value       = azurerm_linux_function_app.main.site_config[0].application_stack[0].python_version
}
