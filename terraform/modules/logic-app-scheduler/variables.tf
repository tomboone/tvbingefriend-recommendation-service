# Variables for Logic App scheduler module

variable "logic_app_name" {
  description = "Name of the Logic App"
  type        = string
}

variable "location" {
  description = "Azure location"
  type        = string
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
}

variable "container_group_id" {
  description = "Resource ID of the container group to manage"
  type        = string
}

variable "container_group_name" {
  description = "Name of the container group"
  type        = string
}

variable "schedule_frequency" {
  description = "Schedule frequency (Day, Week, Month)"
  type        = string
  default     = "Week"
}

variable "schedule_interval" {
  description = "Schedule interval (e.g., 1 for weekly)"
  type        = number
  default     = 1
}

variable "schedule_time" {
  description = "Time to run (24-hour format, e.g., '02:00' for 2 AM)"
  type        = string
  default     = "02:00"
}

variable "schedule_days_of_week" {
  description = "Days of week to run (for weekly schedule, e.g., ['Sunday'])"
  type        = list(string)
  default     = ["Sunday"]
}