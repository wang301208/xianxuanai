/**
 * 数据库配置和Prisma客户端设置
 * 
 * 该模块根据环境配置Prisma数据库客户端的适当日志级别，
 * 并实现单例模式以防止在开发环境中出现多个客户端实例。
 */

import { PrismaClient } from "@prisma/client";
import { env } from "~/env.mjs";

/**
 * Prisma客户端实例的全局容器
 * 用于在开发环境中实现单例模式
 */
const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

/**
 * Prisma数据库客户端实例
 * 
 * 特性：
 * - 开发环境中的单例模式，防止连接问题
 * - 环境特定的日志配置
 * - 自动连接管理
 * 
 * 日志级别：
 * - 开发环境：query, error, warn（用于调试的详细日志）
 * - 生产环境：仅error（为性能优化的最小日志）
 */
export const prisma =
  globalForPrisma.prisma ??
  new PrismaClient({
    log:
      env.NODE_ENV === "development" ? ["query", "error", "warn"] : ["error"],
  });

// 在非生产环境中，将客户端存储在全局范围内，以防止
// 在热重载和开发服务器重启期间出现多个实例
if (env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;
