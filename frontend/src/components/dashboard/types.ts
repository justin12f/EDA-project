import type { LucideIcon } from "lucide-react";
import { Database, Server, FileSpreadsheet, FileJson } from "lucide-react";

export type DataSource = {
  id: string;
  name: string;
  kind: "postgres" | "mysql" | "csv" | "json";
  env: string;
  rows: string;
  icon: LucideIcon;
  status: "live" | "idle" | "syncing";
  table?: string;
};

export const DATA_SOURCES: DataSource[] = [
  {
    id: "prod_pg",
    name: "Production Postgres",
    env: "prod",
    rows: "12.4M",
    kind: "postgres",
    icon: Database,
    status: "live",
    table: "users",
  },
  {
    id: "analytics_mysql",
    name: "Analytics MySQL",
    env: "warehouse",
    rows: "48.1M",
    kind: "mysql",
    icon: Server,
    status: "live",
    table: "events",
  },
  {
    id: "users_csv",
    name: "users_2024.csv",
    env: "upload",
    rows: "482K",
    kind: "csv",
    icon: FileSpreadsheet,
    status: "idle",
    table: "users_2024",
  },
  {
    id: "events_json",
    name: "events.json",
    env: "upload",
    rows: "118K",
    kind: "json",
    icon: FileJson,
    status: "idle",
    table: "events_raw",
  },
];
