# socialgroupdetection

# Example usage

```python
from socialgroupdetection import SGA

# make sure you have BEARER_KEY set as an environment variable or you use bearer_key in constructor

sga = SGA(gwdg_server=True, bearer_key=None) # set bearer key to your gwdg/chatgpt key
response = sga.get_completion(prompt="The teleworker brings only his or her work tools offices are generally equipped and pays for access to the office?")
```

Returns:

 ```json
{
  "explizit": [
    "teleworker"
  ],
  "implizit": [],
  "sonstige": [
    "work tools",
    "offices"
  ]
}
```
