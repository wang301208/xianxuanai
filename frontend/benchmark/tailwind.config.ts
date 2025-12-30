/**
 * AutoGPT 基准测试前端的 Tailwind CSS 配置文件
 * 
 * 本文件配置了基准测试前端应用的 Tailwind CSS 设置，
 * 定义了样式扫描路径、主题扩展和插件配置。
 * 
 * 配置说明:
 * - content: 指定 Tailwind 扫描样式类的文件路径
 * - theme: 主题配置，可以扩展默认设计系统
 * - plugins: Tailwind 插件列表
 * 
 * 技术栈:
 * - Tailwind CSS v3.x
 * - TypeScript 配置
 * - 支持 React/Next.js 项目结构
 */

import type { Config } from "tailwindcss";

export default {
  // 内容扫描配置
  // 指定 Tailwind 需要扫描哪些文件来检测使用的 CSS 类
  // 包括 JavaScript、TypeScript、JSX 和 TSX 文件
  content: ["./src/**/*.{js,ts,jsx,tsx}"],
  
  // 主题配置
  theme: {
    // 主题扩展
    // 在这里可以添加自定义颜色、字体、间距等设计令牌
    // 当前使用默认配置，未进行自定义扩展
    extend: {},
  },
  
  // 插件配置
  // 当前未使用任何 Tailwind 插件
  // 可以在这里添加官方或第三方插件来扩展功能
  plugins: [],
} satisfies Config;
