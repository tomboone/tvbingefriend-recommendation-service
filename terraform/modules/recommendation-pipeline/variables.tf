# Variables for recommendation pipeline module

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
}

variable "location" {
  description = "Azure location"
  type        = string
}

variable "acr_login_server" {
  type = string
}

variable "acr_admin_username" {
  type = string
}

variable "acr_admin_password" {
  type = string
}

variable "image_name" {
  description = "Container image name"
  type        = string
  default     = "recommendation-pipeline"
}

variable "image_tag" {
  description = "Container image tag"
  type        = string
  default     = "latest"
}

variable "cpu_cores" {
  description = "Number of CPU cores"
  type        = number
  default     = 2.0
}

variable "memory_in_gb" {
  description = "Memory in GB"
  type        = number
  default     = 4.0
}

variable "show_service_url" {
  description = "URL for the show service API"
  type        = string
}

variable "storage_container_name" {
  description = "Blob storage container name"
  type        = string
}

variable "storage_primary_connection_string" {
  type = string
  sensitive = true
}

variable "mysql_user" {
  type = string
}

variable "mysql_pwd" {
  type = string
  sensitive = true
}

variable "mysql_fqdn" {
  type = string
}

variable "mysql_db" {
  type = string
}

variable "mysql_charset" {
  type = string
}
