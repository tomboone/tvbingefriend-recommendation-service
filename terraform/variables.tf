# Root-level variables for recommendation service

# Shared infrastructure state
variable "tf_shared_resource_group_name" {
  description = "Resource group name for shared Terraform state"
  type        = string
}

variable "tf_shared_storage_account_name" {
  description = "Storage account name for shared Terraform state"
  type        = string
}

variable "tf_shared_container_name" {
  description = "Container name for shared Terraform state"
  type        = string
}

variable "tf_shared_key" {
  description = "State file key for shared infrastructure"
  type        = string
}

# API configuration
variable "allowed_origins" {
  description = "Allowed CORS origins for the API"
  type        = list(string)
}

# Pipeline configuration
variable "pipeline_image_name" {
  description = "Docker image name for the recommendation pipeline"
  type        = string
  default     = "recommendation-pipeline"
}

variable "pipeline_image_tag" {
  description = "Docker image tag for the recommendation pipeline"
  type        = string
  default     = "latest"
}

variable "pipeline_cpu_cores" {
  description = "CPU cores for the pipeline container"
  type        = number
  default     = 2.0
}

variable "pipeline_memory_in_gb" {
  description = "Memory in GB for the pipeline container"
  type        = number
  default     = 4.0
}

# Service URLs
variable "show_service_url" {
  description = "URL for the show service API"
  type        = string
}

variable "mysql_admin_username" {
  description = "Admin username for MySQL server"
  type = string
}

variable "mysql_admin_password" {
  description = "Admin password for MySQL server"
  type = string
  sensitive = true
}

# Pipeline scheduling configuration
variable "pipeline_schedule_frequency" {
  description = "Schedule frequency (Day, Week, Month)"
  type        = string
  default     = "Week"
}

variable "pipeline_schedule_interval" {
  description = "Schedule interval (e.g., 1 for weekly)"
  type        = number
  default     = 1
}

variable "pipeline_schedule_time" {
  description = "Time to run (24-hour format, e.g., '09:00' for 9 AM UTC / 5 AM ET)"
  type        = string
  default     = "09:00"
}

variable "pipeline_schedule_days_of_week" {
  description = "Days of week to run (for weekly schedule)"
  type        = list(string)
  default     = ["Sunday"]
}

variable "pipeline_notification_email" {
  description = "Email address to send pipeline completion notifications (leave empty to disable)"
  type        = string
  default     = ""
}