# AURA Security Guide

## Overview

AURA implements multiple layers of security to protect the system and user data.

## Security Features

### 1. Rate Limiting

All API endpoints are rate-limited to prevent abuse:
- **Limit**: 100 requests per minute per IP
- **Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`
- **Response**: 429 status when exceeded

### 2. Security Headers

All responses include security headers:
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Content-Security-Policy: default-src 'self'; ...
```

### 3. CORS Protection

Cross-Origin Resource Sharing is configured:
- Only whitelisted origins allowed
- Credentials enabled for authenticated requests
- Limited HTTP methods (GET, POST, PUT, DELETE, OPTIONS)

### 4. Input Sanitization

All user inputs are sanitized:
- HTML/script injection prevention
- Path traversal protection
- Command injection prevention (terminal)
- Length limits enforced

### 5. Terminal Security

The integrated terminal has multiple protections:
- Dangerous command blocking (sudo, rm -rf /, etc.)
- Command pattern matching for suspicious inputs
- Working directory validation
- Timeout limits (30 seconds)
- Shell escape prevention

### 6. Request Size Limits

- Maximum request body: 10MB
- Prevents DoS via large payloads

## Configuration

### Environment Variables

```bash
# Security settings
AURA_SECRET_KEY=your-secret-key
AURA_JWT_SECRET=your-jwt-secret
AURA_ENV=production

# CORS
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### Production Checklist

- [ ] Change default secrets
- [ ] Enable HTTPS only
- [ ] Configure CORS for your domain
- [ ] Set up fail2ban
- [ ] Enable request logging
- [ ] Configure firewall rules
- [ ] Regular security updates
- [ ] Disable debug mode
- [ ] Review API documentation access

## API Security

### Authentication

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.get("/protected")
async def protected_route(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Verify token
    pass
```

### Input Validation

```python
from pydantic import BaseModel, validator

class CreateGoalRequest(BaseModel):
    title: str
    description: str
    
    @validator('title')
    def validate_title(cls, v):
        if len(v) > 200:
            raise ValueError('Title too long')
        return sanitize_input(v)
```

## Web UI Security

### Content Security Policy

The CSP header prevents XSS attacks:
```
default-src 'self';
script-src 'self' 'unsafe-inline';
style-src 'self' 'unsafe-inline' fonts.googleapis.com;
img-src 'self' data: blob:;
connect-src 'self' ws: wss:;
```

### Service Worker Security

- HTTPS required for service workers
- Cache validation on every fetch
- No cross-origin caching

## Best Practices

### 1. Secrets Management

Never commit secrets to version control:
```bash
# Use environment files
cp .env.example .env
# .env is in .gitignore
```

### 2. Database Security

- Use parameterized queries
- Encrypt sensitive data at rest
- Regular backups
- Access controls

### 3. Network Security

```bash
# Firewall rules (ufw)
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 8000/tcp  # API (if exposed)
ufw deny 8000/tcp   # Block direct API access
```

### 4. Monitoring

Set up alerts for:
- Rate limit violations
- Failed authentication attempts
- Suspicious terminal commands
- Error spikes

## Incident Response

### Suspicious Activity

1. Check logs:
```bash
docker-compose logs api | grep "WARN\|ERROR"
```

2. Review rate limit hits:
```bash
docker-compose logs api | grep "rate_limit_exceeded"
```

3. Check terminal blocks:
```bash
docker-compose logs api | grep "blocked_command"
```

### Emergency Response

If compromised:
1. Revoke all sessions
2. Rotate secrets
3. Review access logs
4. Update firewall rules
5. Notify users if data exposed

## Penetration Testing

Recommended tools:
- OWASP ZAP
- Burp Suite
- Nikto
- SQLMap

### Common Tests

```bash
# Test for SQL injection
sqlmap -u "http://localhost:8000/api/goals?id=1"

# Test for XSS
curl -X POST http://localhost:8000/api/chat \
  -d '{"message": "<script>alert(1)</script>"}'

# Test rate limiting
for i in {1..150}; do curl http://localhost:8000/api/health; done
```

## Compliance

### GDPR

- Data minimization
- Right to deletion
- Consent management
- Breach notification

### SOC 2

- Access controls
- Audit logging
- Encryption
- Monitoring

## Security Updates

Stay updated:
```bash
# Check for vulnerabilities
npm audit
pip safety check

# Update dependencies
npm update
pip install -U -r requirements.txt
```

## Reporting Issues

Report security vulnerabilities to: security@yourdomain.com

Include:
- Description
- Steps to reproduce
- Impact assessment
- Suggested fix
