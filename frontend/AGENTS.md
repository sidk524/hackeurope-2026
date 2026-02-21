# Repository Guidelines

## Project Structure & Module Organization
- `src/app` contains the Next.js App Router entry points (`layout.tsx`, `page.tsx`) plus route-level assets like `globals.css`.
- `src/app/components` hosts page-scoped React components (e.g., `ThreeScene.tsx`).
- `src/components` holds shared UI primitives (currently `ui/button.tsx`, shadcn-style).
- `src/lib` is for reusable utilities (`utils.ts` for Tailwind class merging).
- `public` contains static assets served at the site root.

## Build, Test, and Development Commands
- `npm run dev` starts the local dev server at `http://localhost:3000`.
- `npm run build` creates the production build with Next.js.
- `npm run start` serves the production build locally.

## Coding Style & Naming Conventions
- TypeScript is enabled with `strict` settings (`tsconfig.json`). Prefer typed props and avoid `any` unless justified.
- Use Tailwind CSS v4 + shadcn utilities; prefer `clsx` + `tailwind-merge` for class composition (`src/lib/utils.ts`).
- React components use `PascalCase` file names (e.g., `ProjectsClient.tsx`), hooks/utilities use `camelCase`.
- Indentation is 2 spaces in config files and 2 spaces for TypeScript/TSX.

## Testing Guidelines
- No test runner or test scripts are configured yet. If you add tests, include the framework and how to run it in this file and add a script to `package.json`.

## Commit & Pull Request Guidelines
- Commit history uses short, descriptive, sentence-like messages (no Conventional Commits observed). Keep messages concise and specific to the change.
- PRs should include:
  - A short summary of changes and user impact.
  - Screenshots or recordings for UI changes.
  - Notes on local testing (which scripts you ran).

## Configuration & Assets
- Styling is centralized in `src/app/globals.css` using Tailwind layers and CSS variables.
- Static assets should go in `public/` and be referenced with absolute paths like `/logo.svg`.
