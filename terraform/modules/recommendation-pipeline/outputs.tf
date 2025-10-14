# Outputs for recommendation pipeline module

output "container_group_id" {
  description = "ID of the container group"
  value       = azurerm_container_group.pipeline.id
}

output "container_group_name" {
  description = "Name of the container group"
  value       = azurerm_container_group.pipeline.name
}

output "container_group_location" {
  description = "Location of the container group"
  value       = azurerm_container_group.pipeline.location
}
