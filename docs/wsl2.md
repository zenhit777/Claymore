# WSL2 notes

Suggested starting point for `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=96GB
processors=16
swap=16GB
```

Restart WSL after changing the file (`wsl --shutdown`). Keep swap out of the model-weight tier; use it only as an emergency system mechanism.
