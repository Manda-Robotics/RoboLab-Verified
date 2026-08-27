#!/bin/bash
# Stop this pod from inside itself (GPU billing ends; /workspace persists).
# Uses the pod-scoped RUNPOD_API_KEY via GraphQL — note runpodctl subcommands
# are NOT authorized with the pod-scoped key, only direct GraphQL is.
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_POD_ID=|^RUNPOD_API_KEY=" | xargs)
curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}"
