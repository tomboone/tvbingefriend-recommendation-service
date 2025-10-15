# Container monitoring module with email notifications
# Monitors container completion status and sends email alerts

# Action Group for email notifications
resource "azurerm_monitor_action_group" "pipeline_notifications" {
  count = var.notification_email != "" ? 1 : 0

  name                = "${var.container_group_name}-notifications"
  resource_group_name = var.resource_group_name
  short_name          = "pipeline"

  email_receiver {
    name                    = "Pipeline Admin"
    email_address          = var.notification_email
    use_common_alert_schema = true
  }

  tags = {
    purpose = "recommendation-pipeline-monitoring"
  }
}

# Activity Log Alert for container success (Succeeded state)
resource "azurerm_monitor_activity_log_alert" "container_succeeded" {
  count = var.notification_email != "" ? 1 : 0

  name                = "${var.container_group_name}-succeeded"
  resource_group_name = var.resource_group_name
  scopes              = [var.container_group_id]
  description         = "Alert when recommendation pipeline container completes successfully"

  criteria {
    category       = "Administrative"
    operation_name = "Microsoft.ContainerInstance/containerGroups/write"
    level          = "Informational"

    resource_type = "Microsoft.ContainerInstance/containerGroups"
    resource_id   = var.container_group_id
  }

  action {
    action_group_id = azurerm_monitor_action_group.pipeline_notifications[0].id
  }

  tags = {
    purpose = "recommendation-pipeline-monitoring"
  }
}

# Activity Log Alert for container failure
resource "azurerm_monitor_activity_log_alert" "container_failed" {
  count = var.notification_email != "" ? 1 : 0

  name                = "${var.container_group_name}-failed"
  resource_group_name = var.resource_group_name
  scopes              = [var.container_group_id]
  description         = "Alert when recommendation pipeline container fails"

  criteria {
    category       = "Administrative"
    operation_name = "Microsoft.ContainerInstance/containerGroups/write"
    level          = "Error"

    resource_type = "Microsoft.ContainerInstance/containerGroups"
    resource_id   = var.container_group_id
  }

  action {
    action_group_id = azurerm_monitor_action_group.pipeline_notifications[0].id
  }

  tags = {
    purpose = "recommendation-pipeline-monitoring"
  }
}
