# Outputs for Logic App scheduler module

output "logic_app_id" {
  description = "ID of the Logic App"
  value       = data.azurerm_logic_app_workflow.scheduler.id
}

output "logic_app_name" {
  description = "Name of the Logic App"
  value       = data.azurerm_logic_app_workflow.scheduler.name
}

output "logic_app_identity_principal_id" {
  description = "Principal ID of the Logic App managed identity"
  # IDE warning about "Unresolved reference" is expected - the output structure is defined in the ARM template
  value       = jsondecode(azurerm_resource_group_template_deployment.scheduler.output_content).principalId.value
}

output "logic_app_access_endpoint" {
  description = "Access endpoint of the Logic App"
  value       = data.azurerm_logic_app_workflow.scheduler.access_endpoint
}
