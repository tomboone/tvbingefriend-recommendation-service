# Logic App for scheduled container execution
# This creates a Logic App that runs the recommendation pipeline on a schedule
# Using ARM template to support managed identity authentication for HTTP actions

locals {
  # Parse schedule time (e.g., "02:00" -> hour=2, minute=0)
  schedule_parts = split(":", var.schedule_time)
  schedule_hour  = local.schedule_parts[0]
  schedule_minute = local.schedule_parts[1]

  # Build schedule configuration for recurrence trigger
  schedule_config = var.schedule_frequency == "Week" ? {
    hours   = [tonumber(local.schedule_hour)]
    minutes = [tonumber(local.schedule_minute)]
    weekDays = var.schedule_days_of_week
  } : {
    hours   = [tonumber(local.schedule_hour)]
    minutes = [tonumber(local.schedule_minute)]
  }
}

# Deploy Logic App with complete workflow definition via ARM template
# This is necessary to configure managed identity authentication for HTTP actions
resource "azurerm_resource_group_template_deployment" "scheduler" {
  name                = "logic-app-scheduler-deployment"
  resource_group_name = var.resource_group_name
  deployment_mode     = "Incremental"

  template_content = jsonencode({
    "$schema" : "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
    "contentVersion" : "1.0.0.0",
    "resources" : [
      {
        "type" : "Microsoft.Logic/workflows",
        "apiVersion" : "2019-05-01",
        "name" : var.logic_app_name,
        "location" : var.location,
        "identity" : {
          "type" : "SystemAssigned"
        },
        "tags" : {
          "purpose" : "recommendation-pipeline-scheduler"
        },
        "properties" : {
          "definition" : {
            "$schema" : "https://schema.management.azure.com/providers/Microsoft.Logic/schemas/2016-06-01/workflowdefinition.json#",
            "contentVersion" : "1.0.0.0",
            "triggers" : {
              "RecurrenceSchedule" : {
                "type" : "Recurrence",
                "recurrence" : {
                  "frequency" : var.schedule_frequency,
                  "interval" : var.schedule_interval,
                  "timeZone" : "UTC",
                  "schedule" : local.schedule_config
                }
              }
            },
            "actions" : {
              "Start_Container_Group" : {
                "type" : "Http",
                "inputs" : {
                  "method" : "POST",
                  "uri" : "https://management.azure.com${var.container_group_id}/start?api-version=2021-09-01",
                  "authentication" : {
                    "type" : "ManagedServiceIdentity",
                    "audience" : "https://management.azure.com/"
                  }
                },
                "runAfter" : {}
              }
            }
          }
        }
      }
    ],
    "outputs" : {
      "logicAppId" : {
        "type" : "string",
        "value" : "[resourceId('Microsoft.Logic/workflows', '${var.logic_app_name}')]"
      },
      "principalId" : {
        "type" : "string",
        "value" : "[reference(resourceId('Microsoft.Logic/workflows', '${var.logic_app_name}'), '2019-05-01', 'Full').identity.principalId]"
      }
    }
  })
}

# Get the Logic App details from the deployment outputs
data "azurerm_logic_app_workflow" "scheduler" {
  name                = var.logic_app_name
  resource_group_name = var.resource_group_name

  depends_on = [azurerm_resource_group_template_deployment.scheduler]
}

# Role assignment to allow Logic App to manage Container Instances
resource "azurerm_role_assignment" "container_contributor" {
  scope                = var.container_group_id
  role_definition_name = "Contributor"
  # Extract principal ID from ARM template deployment outputs
  # IDE warning about "Unresolved reference" is expected - the output structure is defined in the ARM template
  # noinspection HILUnresolvedReference
  principal_id         = jsondecode(azurerm_resource_group_template_deployment.scheduler.output_content).principalId.value
}
