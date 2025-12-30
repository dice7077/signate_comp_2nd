# signate_comp_2nd

## Git hooks

This repo ships shared git hooks under `scripts/git-hooks`. They enforce basic
workflow rules—currently blocking direct commits to `main`.

Install them once per clone:

```
git config core.hooksPath scripts/git-hooks
```

After that, any `git commit` while on `main` will abort with a friendly message.
Create a feature branch (e.g. `git checkout -b feat/my-change`) and commit
there instead.