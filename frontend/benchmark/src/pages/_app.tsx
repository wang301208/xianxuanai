import { useCallback, useMemo, useState } from "react";
import type { AppProps } from "next/app";
import "~/styles/globals.css";
import "@fortawesome/fontawesome-svg-core/styles.css";

import Layout from "../components/layout/Layout";
import {
  AppStatus,
  AppStatusProvider,
} from "../components/layout/AppStatusContext";

const MyApp = ({ Component, pageProps }: AppProps) => {
  const [status, setStatus] = useState<AppStatus>({
    agentState: "Idle",
    learningEnabled: true,
    lastSync: new Date().toLocaleTimeString(),
  });

  const refreshStatus = useCallback(async () => {
    try {
      if (typeof window !== "undefined") {
        const response = await fetch("/api/status", { method: "GET" });
        if (response.ok) {
          const payload = await response.json();
          setStatus({
            agentState: payload.agentState ?? "Running",
            learningEnabled:
              typeof payload.learningEnabled === "boolean"
                ? payload.learningEnabled
                : true,
            lastSync:
              payload.lastSync ?? new Date().toLocaleTimeString(),
          });
          return;
        }
      }
    } catch (error) {
      console.warn("Falling back to mock status update", error);
    }
    setStatus((previous) => ({
      ...previous,
      lastSync: new Date().toLocaleTimeString(),
    }));
  }, []);

  const stopCurrentRun = useCallback(async () => {
    try {
      if (typeof window !== "undefined") {
        await fetch("/api/stop", { method: "POST" });
      }
    } catch (error) {
      console.warn("Stop run endpoint not available", error);
    } finally {
      setStatus((previous) => ({
        ...previous,
        agentState: "Idle",
        lastSync: new Date().toLocaleTimeString(),
      }));
    }
  }, []);

  const value = useMemo(
    () => ({
      status,
      refreshStatus,
      stopCurrentRun,
    }),
    [status, refreshStatus, stopCurrentRun]
  );

  return (
    <AppStatusProvider value={value}>
      <Layout>
        <Component {...pageProps} />
      </Layout>
    </AppStatusProvider>
  );
};

export default MyApp;
