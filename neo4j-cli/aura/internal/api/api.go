// Copyright (c) "Neo4j"
// Neo4j Sweden AB [http://neo4j.com]

package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"

	"github.com/neo4j/cli/common/clicfg"
)

const userAgent = "Neo4jCLI/%s"

type Grant struct {
	AccessToken string `json:"access_token"`
	ExpiresIn   int64  `json:"expires_in"`
}

type AuraApiVersion string

const (
	AuraApiVersion1 AuraApiVersion = "1"
	AuraApiVersion2 AuraApiVersion = "2"
)

type RequestConfig struct {
	Version     AuraApiVersion
	Method      string
	PostBody    map[string]any
	QueryParams map[string]string
}

func MakeRequest(cfg *clicfg.Config, path string, config *RequestConfig) (responseBody []byte, statusCode int, err error) {
	client := http.Client{}
	method := config.Method
	if method == "" {
		return nil, 0, fmt.Errorf("method not set in request %s", path)
	}

	body, err := createBody(config.PostBody)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to create request body: %w", err)
	}

	baseUrl := cfg.Aura.BaseUrl()
	if config.Version == "" {
		config.Version = AuraApiVersion1
	}
	versionPath := getVersionPath(cfg, config.Version)

	u, err := url.ParseRequestURI(baseUrl)
	if err != nil {
		return nil, 0, fmt.Errorf("invalid base URL %q: %w", baseUrl, err)
	}
	u = u.JoinPath(versionPath)
	u = u.JoinPath(path)

	addQueryParams(u, config.QueryParams)

	urlString := u.String()
	req, err := http.NewRequest(method, urlString, body)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	credential, err := cfg.Credentials.Aura.GetDefault()
	if err != nil {
		return nil, 0, err
	}

	req.Header, err = getHeaders(credential, cfg)
	if err != nil {
		return nil, 0, err
	}

	res, err := client.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("HTTP request failed: %w", err)
	}

	defer res.Body.Close()

	if IsSuccessful(res.StatusCode) {
		responseBody, err = io.ReadAll(res.Body)
		if err != nil {
			return nil, res.StatusCode, fmt.Errorf("failed to read response body: %w", err)
		}

		return responseBody, res.StatusCode, nil
	}

	return responseBody, res.StatusCode, handleResponseError(res, credential, cfg)
}

func getVersionPath(cfg *clicfg.Config, version AuraApiVersion) string {
	betaEnabled := cfg.Aura.AuraBetaEnabled()

	switch version {
	case AuraApiVersion1:
		if betaEnabled {
			return cfg.Aura.BetaPathV1()
		}
		return "v1"
	case AuraApiVersion2:
		if betaEnabled {
			return cfg.Aura.BetaPathV2()
		}
		return "v2"
	default:
		return "v1"
	}
}

func createBody(data map[string]any) (io.Reader, error) {
	if data == nil {
		return nil, nil
	}

	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}

	return bytes.NewBuffer(jsonData), nil
}

func addQueryParams(u *url.URL, params map[string]string) {
	if params != nil {
		q := u.Query()
		for key, val := range params {
			q.Add(key, val)
		}
		u.RawQuery = q.Encode()
	}
}

// Checks status code is 2xx
func IsSuccessful(statusCode int) bool {
	return statusCode >= 200 && statusCode <= 299
}
