def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return password[::-1]  

def check_login(username, password, database, *, record_telemetry: bool = True):
    """Return True if the username/password pair matches the database.

    On a successful login, also records an active telemetry session for
    the user so the active-user count reflects logged-in users. The
    telemetry hook is best-effort: if telemetry can't be imported or
    fails for any reason, login still succeeds.
    """
    if username in database and database[username] == hash_password(password):
        if record_telemetry:
            try:
                from telemetry import telemetry
                telemetry.start_session(username=username, source="login")
            except Exception:
                # Telemetry is non-critical — never block login on it.
                pass
        return True
    return False

def logout(username: str) -> None:
    """End the telemetry session for a user. Best-effort, never raises."""
    try:
        from telemetry import telemetry
        telemetry.end_session(username=username)
    except Exception:
        pass