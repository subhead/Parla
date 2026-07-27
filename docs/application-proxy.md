# Application Proxy

Application Proxy controls how Parla reaches network services.
It applies to model and VAD downloads, cloud batch transcription, LLM enhancement, and streaming transcription.

## Proxy modes

Choose proxy mode in Parla's network settings.

### Disabled Direct

`Disabled Direct` sends requests directly.
Parla does not use Windows system proxy settings, environment proxy settings, or an Application Proxy in this mode.

### Explicit proxy URL

Enter an explicit URL to route requests through one proxy.
Supported schemes are `http://`, `https://`, and `socks5://`.
For example:

```text
http://proxy.example.com:8080
https://proxy.example.com:8443
socks5://proxy.example.com:1080
```

Do not put credentials in the URL.
Store proxy username and password in Parla's separate credential fields so they remain separate from routing configuration.

### Windows System proxy

Enable Application Proxy and leave its URL blank to use the Windows System proxy for each new request.
This mode follows Windows proxy configuration, including PAC and WPAD discovery and integrated authentication where supported by Windows.

When Windows selects a direct route through PAC/WPAD, Parla sends the request directly.
When Windows selects a proxy, Parla uses that proxy.

## No-Proxy rule

No-Proxy Entries apply only when an explicit proxy URL is configured.
Matching destinations bypass that explicit Application Proxy and connect directly.
Entries use `NO_PROXY`-compatible matching, including exact hosts, wildcard domains, CIDR ranges, ports, leading dots, and IP variants.
No destinations bypass an explicit proxy unless they match a listed entry.

When using Windows System proxy mode, Parla defers bypass decisions to Windows and its PAC/WPAD configuration.
No-Proxy Entries do not override Windows system proxy decisions in that mode.

## What this covers

Application Proxy uses one routing policy for Parla outbound HTTP(S) traffic across these operations:

- Downloading local transcription models and VAD models.
- Cloud batch transcription.
- LLM enhancement requests.
- Streaming transcription requests and connections.

Local inference and local services do not become network-proxied merely because Application Proxy is enabled.

## Manual Windows laptop validation matrix

The following matrix is a manual validation checklist for a Windows laptop connected to a corporate network.
Every row is pending manual Windows corporate validation and must not be read as an executed or verified result.
Run each row with Windows System proxy mode enabled and the URL blank.
Repeat with a PAC URL configured and with WPAD discovery where each is available.

| Route configuration | Scope | Expected behavior | Execution result | Evidence |
|---|---|---|---|---|
| Windows System proxy with PAC | Model and VAD downloads; cloud HTTP batch; LLM enhancement; streaming transcription | Each operation uses PAC-selected route without direct-route fallback | Pending manual Windows corporate validation | Parla success or error details, Windows proxy/PAC diagnostics, and corporate proxy logs where available |
| Windows System proxy with WPAD | Model and VAD downloads; cloud HTTP batch; LLM enhancement; streaming transcription | Each operation uses WPAD-selected route without direct-route fallback | Pending manual Windows corporate validation | Parla success or error details, Windows proxy/PAC diagnostics, and corporate proxy logs where available |
| Windows System proxy with PAC/WPAD direct result | Model and VAD downloads; cloud HTTP batch; LLM enhancement; streaming transcription | Each operation connects directly when PAC/WPAD selects a direct route | Pending manual Windows corporate validation | PAC/WPAD decision shows direct, no proxy request appears, and Parla success or error details are recorded |

Expected evidence should come from both sides where possible: Parla success or error details, Windows proxy/PAC diagnostics, and corporate proxy logs.
Confirm that model and VAD downloads, cloud batch traffic, LLM traffic, and streaming traffic each follow the selected Windows route.

This matrix is manual Windows laptop validation, not proof of compatibility with every corporate proxy.
Real corporate validation remains limited by proxy product, PAC/WPAD rules, authentication policy, TLS inspection, firewall policy, and streaming protocol support.
