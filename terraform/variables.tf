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