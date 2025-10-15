# Logic App for scheduled container execution
# This creates a Logic App that runs the recommendation pipeline on a schedule

locals {
  # Parse schedule time (e.g., "02:00" -> hour=2, minute=0)
  schedule_parts = split(":", var.schedule_time)
  schedule_hour  = local.schedule_parts[0]
  schedule_minute = local.schedule_parts[1]

  # Convert days of week list to JSON array string
  schedule_days_of_week_json = jsonencode(var.schedule_days_of_week)
}

resource "azurerm_logic_app_workflow" "scheduler" {
  name                = var.logic_app_name
  location            = var.location
  resource_group_name = var.resource_group_name

  identity {
    type = "SystemAssigned"
  }

  tags = {
    purpose = "recommendation-pipeline-scheduler"
  }
}

# Role assignment to allow Logic App to manage Container Instances
resource "azurerm_role_assignment" "container_contributor" {
  scope                = var.container_group_id
  role_definition_name = "Contributor"
  principal_id         = azurerm_logic_app_workflow.scheduler.identity[0].principal_id
}

# Deploy the trigger
resource "azurerm_logic_app_trigger_recurrence" "schedule" {
  name         = "RecurrenceSchedule"
  logic_app_id = azurerm_logic_app_workflow.scheduler.id
  frequency    = var.schedule_frequency
  interval     = var.schedule_interval
  time_zone    = "UTC"

  schedule {
    at_these_hours   = [tonumber(local.schedule_hour)]
    at_these_minutes = [tonumber(local.schedule_minute)]
    on_these_days    = var.schedule_frequency == "Week" ? var.schedule_days_of_week : []
  }
}

# Deploy the action to start the container group
# Note: Container will auto-execute pipeline on start, then stop due to OnFailure restart policy
resource "azurerm_logic_app_action_http" "start_container" {
  name         = "Start_Container_Group"
  logic_app_id = azurerm_logic_app_workflow.scheduler.id

  method = "POST"
  uri    = "https://management.azure.com${var.container_group_id}/start?api-version=2021-09-01"

  headers = {
    "Content-Type" = "application/json"
  }

  depends_on = [
    azurerm_logic_app_trigger_recurrence.schedule,
    azurerm_role_assignment.container_contributor
  ]
}
