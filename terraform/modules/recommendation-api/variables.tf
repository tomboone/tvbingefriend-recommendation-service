# Variables for recommendation API module
variable "app_service_plan_name" {
  type = string
}

variable "app_service_plan_resource_group" {
  type = string
}

variable "log_analytics_workspace_name" {
  type = string
}

variable "log_analytics_workspace_resource_group_name" {
  type = string
}

variable "service_name" {
  type = string
}

variable "resource_group_name" {
  type = string
}

variable "location" {
  type = string
}

variable "storage_account_name" {
  type = string
}

variable "storage_account_access_key" {
  type      = string
  sensitive = true
}

variable "allowed_origins" {
  type = list(string)
}

variable "mysql_fqdn" {
  type = string
}

variable "mysql_user" {
  type = string
}

variable "mysql_pwd" {
  type      = string
  sensitive = true
}

variable "mysql_db" {
  type = string
}

variable "mysql_charset" {
  type = string
}
