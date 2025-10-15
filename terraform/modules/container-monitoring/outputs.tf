# Outputs for container monitoring module

output "action_group_id" {
  description = "ID of the action group for pipeline notifications"
  value       = length(azurerm_monitor_action_group.pipeline_notifications) > 0 ? azurerm_monitor_action_group.pipeline_notifications[0].id : null
}

output "action_group_name" {
  description = "Name of the action group"
  value       = length(azurerm_monitor_action_group.pipeline_notifications) > 0 ? azurerm_monitor_action_group.pipeline_notifications[0].name : null
}
