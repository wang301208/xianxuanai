import { createContext, useContext } from "react";

export interface AppStatus {
  agentState: string;
  learningEnabled: boolean;
  lastSync: string;
}

export interface AppStatusContextValue {
  status: AppStatus;
  refreshStatus: () => Promise<void> | void;
  stopCurrentRun: () => Promise<void> | void;
}

const defaultContext: AppStatusContextValue = {
  status: {
    agentState: "Idle",
    learningEnabled: true,
    lastSync: new Date().toISOString(),
  },
  refreshStatus: async () => {},
  stopCurrentRun: async () => {},
};

const AppStatusContext =
  createContext<AppStatusContextValue>(defaultContext);

export const AppStatusProvider = AppStatusContext.Provider;

export const useAppStatus = () => {
  return useContext(AppStatusContext);
};

