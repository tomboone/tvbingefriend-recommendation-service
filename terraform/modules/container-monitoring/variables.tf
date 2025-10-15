# Variables for container monitoring module

variable "container_group_name" {
  description = "Name of the container group to monitor"
  type        = string
}

variable "container_group_id" {
  description = "Resource ID of the container group to monitor"
  type        = string
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
}

variable "notification_email" {
  description = "Email address to send alerts to (leave empty to disable notifications)"
  type        = string
  default     = ""
}
