"""
PyMySQL shim — falls back to PyMySQL if mysqlclient isn't available.
Allows the project to run on Windows without compiling mysqlclient.
"""
try:
    import MySQLdb  # noqa: F401 - mysqlclient is available, use it.
except ImportError:
    try:
        import pymysql

        pymysql.install_as_MySQLdb()
    except ImportError:
        # Neither driver installed. Django will raise a clear error at DB connect time.
        pass
