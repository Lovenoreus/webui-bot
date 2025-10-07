# SSL Configuration Guide for Vanna Database Connections

This guide explains how to configure SSL connections for all supported database types in the Vanna training system.

## Overview

The system automatically handles SSL certificate verification issues by implementing fallback mechanisms for all database types. When a certificate verification error occurs, the system will retry the connection with appropriate SSL bypass settings.

## SSL Error Handling

The system detects the following SSL-related errors and automatically applies fallbacks:
- `certificate verify failed`
- `self-signed certificate in certificate chain`
- `SSL connection errors`
- `TLS handshake errors`
- `CERTIFICATE_VERIFY_FAILED`

## Database-Specific SSL Configuration

### PostgreSQL

**Configuration Options (in config.json):**
```json
"postgresql": {
  "enabled": false,
  "ssl_mode": "require",
  "ssl_fallback_mode": "require",
  "ssl_verify_cert": false
}
```

**SSL Modes:**
- `disable`: No SSL
- `allow`: Try SSL, fallback to non-SSL
- `prefer`: Try SSL first (default)
- `require`: Require SSL but don't verify certificates
- `verify-ca`: Require SSL and verify CA
- `verify-full`: Require SSL and verify hostname

**Automatic Fallback:**
If certificate verification fails, the system automatically retries with the `ssl_fallback_mode` setting.

### MySQL

**Configuration Options (in config.json):**
```json
"mysql": {
  "enabled": false,
  "charset": "utf8mb4",
  "ssl_verify_cert": false,
  "ssl_verify_identity": false,
  "ssl_disabled": false
}
```

**SSL Options:**
- `ssl_verify_cert`: Verify server certificate (false = skip verification)
- `ssl_verify_identity`: Verify server identity (false = skip verification)
- `ssl_disabled`: Completely disable SSL (false = use SSL)

**Automatic Fallback:**
If certificate verification fails, the system automatically retries with `ssl_verify_cert=false` and `ssl_verify_identity=false`.

### Microsoft SQL Server

**Configuration Options (in config.json):**
```json
"mssql": {
  "enabled": false,
  "driver": "ODBC Driver 17 for SQL Server",
  "trusted_connection": false,
  "trust_server_certificate": true,
  "encrypt": true
}
```

**SSL Options:**
- `trust_server_certificate`: Accept self-signed certificates (true = accept)
- `encrypt`: Use encryption (true = use SSL/TLS)
- `trusted_connection`: Use Windows authentication

**Automatic Fallback:**
If certificate verification fails, the system automatically retries with `TrustServerCertificate=yes` and `Encrypt=yes`.

### SQLite

SQLite connections are local file-based and do not use SSL, so no SSL configuration is needed.

## Environment Variables

You can override database connection settings using environment variables:

```bash
# Database connection details
export DB_HOST="your-database-host"
export DB_DATABASE="your-database-name"
export DB_USERNAME="your-username"
export DB_PASSWORD="your-password"
export DB_PORT="custom-port"

# SSL-specific overrides
export POSTGRES_SSLMODE="require"
export MYSQL_SSL_DISABLED="false"
export MSSQL_TRUST_CERT="true"
```

## Docker Environment

When running in Docker containers, SSL certificate issues are common. The system includes:

1. **Certificate Chain Detection**: Automatically detects self-signed certificate errors
2. **Container-Safe Defaults**: Uses SSL settings that work in containerized environments
3. **Debug Logging**: Provides clear feedback when SSL fallbacks are triggered

## Common SSL Scenarios

### 1. Development Environment
```json
"postgresql": {"ssl_mode": "require"},
"mysql": {"ssl_verify_cert": false},
"mssql": {"trust_server_certificate": true}
```

### 2. Production with Proper Certificates
```json
"postgresql": {"ssl_mode": "verify-full"},
"mysql": {"ssl_verify_cert": true, "ssl_verify_identity": true},
"mssql": {"trust_server_certificate": false, "encrypt": true}
```

### 3. Local Testing (SSL Disabled)
```json
"postgresql": {"ssl_mode": "disable"},
"mysql": {"ssl_disabled": true},
"mssql": {"encrypt": false}
```

## Troubleshooting

### Common Error Messages and Solutions

1. **"certificate verify failed: self-signed certificate"**
   - **Solution**: System automatically retries with certificate verification disabled
   - **Manual Fix**: Set appropriate bypass flags in config.json

2. **"SSL connection has been closed unexpectedly"**
   - **Solution**: Check network connectivity and SSL configuration
   - **Manual Fix**: Try disabling SSL or using different SSL mode

3. **"SSL is required but the server doesn't support it"**
   - **Solution**: Set SSL mode to "prefer" or "disable"
   - **Manual Fix**: Update server to support SSL or disable SSL requirement

### Debug Information

The system provides debug output when SSL issues occur:
```
[VANNA DEBUG] Detected SSL certificate error for postgresql: certificate verify failed
[VANNA DEBUG] SSL certificate verification failed, retrying with fallback SSL mode
```

## Security Considerations

### Development vs Production

- **Development**: It's acceptable to bypass certificate verification for local testing
- **Production**: Always use proper SSL certificates when possible
- **Docker**: Container environments often require certificate verification bypass

### Best Practices

1. **Use proper certificates in production**
2. **Enable SSL verification when certificates are trusted**
3. **Monitor debug logs for SSL fallback usage**
4. **Regularly update SSL certificates**
5. **Use environment variables for sensitive configuration**

## Advanced Configuration

For advanced SSL configuration, you can:

1. **Provide custom certificate files**
2. **Configure specific cipher suites**
3. **Set up mutual TLS authentication**
4. **Use certificate bundles**

These advanced features may require additional configuration beyond the automatic SSL handling provided by the system.