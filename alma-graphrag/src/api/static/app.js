const state = {
  network: null,
  overview: null,
  cy: null,
  map: null,
  mapMarkers: [],
  mapInfoWindow: null,
  googleMapsPromise: null,
  ingestJobId: null,
  ingestPolling: null,
  seenLogCount: 0,
};

const els = {
  cityInput: document.getElementById("cityInput"),
  limitInput: document.getElementById("limitInput"),
  applyBtn: document.getElementById("applyBtn"),
  refreshBtn: document.getElementById("refreshBtn"),
  clearBtn: document.getElementById("clearBtn"),
  hotelMap: document.getElementById("hotelMap"),
  mapStatus: document.getElementById("mapStatus"),
  confirmModal: document.getElementById("confirmModal"),
  confirmInput: document.getElementById("confirmInput"),
  confirmOkBtn: document.getElementById("confirmOkBtn"),
  confirmCancelBtn: document.getElementById("confirmCancelBtn"),
  progressText: document.getElementById("progressText"),
  progressFill: document.getElementById("progressFill"),
  debugLog: document.getElementById("debugLog"),
  queryForm: document.getElementById("queryForm"),
  questionInput: document.getElementById("questionInput"),
  answerText: document.getElementById("answerText"),
  contextText: document.getElementById("contextText"),
  statsGrid: document.getElementById("statsGrid"),
  labelList: document.getElementById("labelList"),
  relList: document.getElementById("relList"),
  nodesTableBody: document.querySelector("#nodesTable tbody"),
  nodeMeta: document.getElementById("nodeMeta"),
  nodeProps: document.getElementById("nodeProps"),
  neighborList: document.getElementById("neighborList"),
  graphLegend: document.getElementById("graphLegend"),
  toast: document.getElementById("toast"),
};

const labelPalette = {
  Hotel: "#45e0c8",
  City: "#f5b700",
  Event: "#ff8f70",
  NewsSignal: "#7fa8ff",
  Amenity: "#9de06a",
  Location: "#b68bff",
};

function toast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.style.borderColor = isError ? "rgba(255,110,95,0.6)" : "rgba(69,224,200,0.5)";
  els.toast.classList.add("show");
  window.setTimeout(() => els.toast.classList.remove("show"), 2600);
}

function addDebugLog(message, ts) {
  const stamp = ts
    ? new Date(ts * 1000).toLocaleTimeString()
    : new Date().toLocaleTimeString();
  const line = `[${stamp}] ${message}`;
  const current = (els.debugLog.textContent || "").split("\n").slice(-60);
  current.push(line);
  els.debugLog.textContent = current.join("\n");
  els.debugLog.scrollTop = els.debugLog.scrollHeight;
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) {
    return null;
  }
  const total = Math.max(0, Number(seconds));
  const mins = Math.floor(total / 60);
  const secs = Math.floor(total % 60);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function setProgress(percent, text, meta = {}) {
  const safe = Math.max(0, Math.min(100, Number(percent || 0)));
  els.progressFill.style.width = `${safe}%`;

  const parts = [`${safe}%`, text || "Working"];
  if (meta.step_index && meta.step_total) {
    parts.push(`Step ${meta.step_index}/${meta.step_total}`);
  }
  if (meta.step_progress !== null && meta.step_progress !== undefined) {
    parts.push(`Step progress ${Number(meta.step_progress)}%`);
  }
  const eta = formatDuration(meta.eta_seconds);
  if (eta && safe < 100) {
    parts.push(`ETA ${eta}`);
  }
  els.progressText.textContent = parts.join(" • ");
}

function setMapStatus(text) {
  if (els.mapStatus) {
    els.mapStatus.textContent = text;
  }
}

function clearMapMarkers() {
  for (const marker of state.mapMarkers) {
    marker.setMap(null);
  }
  state.mapMarkers = [];
}

function loadGoogleMaps(apiKey) {
  if (window.google && window.google.maps) {
    return Promise.resolve();
  }
  if (state.googleMapsPromise) {
    return state.googleMapsPromise;
  }

  state.googleMapsPromise = new Promise((resolve, reject) => {
    const timeoutId = window.setTimeout(() => {
      reject(new Error("Google Maps load timeout"));
    }, 8000);

    const callbackName = `__almaGoogleMapInit_${Date.now()}`;
    window[callbackName] = () => {
      window.clearTimeout(timeoutId);
      delete window[callbackName];
      resolve();
    };

    const script = document.createElement("script");
    script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}&libraries=places&callback=${callbackName}`;
    script.async = true;
    script.defer = true;
    script.onerror = () => {
      window.clearTimeout(timeoutId);
      reject(new Error("Failed to load Google Maps script"));
    };
    document.head.appendChild(script);
  });

  return state.googleMapsPromise;
}

async function initHotelMap() {
  if (state.map) {
    return true;
  }

  try {
    const cfg = await request("/client/config");
    const apiKey = cfg.google_maps_api_key;
    if (!apiKey) {
      setMapStatus("Google Maps API key missing in server config");
      addDebugLog("map disabled: GOOGLE_MAPS_API_KEY not configured");
      return false;
    }

    await loadGoogleMaps(apiKey);

    state.map = new window.google.maps.Map(els.hotelMap, {
      center: { lat: 7.8731, lng: 80.7718 },
      zoom: 7,
      mapTypeControl: false,
      fullscreenControl: false,
      streetViewControl: false,
      styles: [
        { elementType: "geometry", stylers: [{ color: "#0b0908" }] },
        { elementType: "labels.text.stroke", stylers: [{ color: "#0b0908" }] },
        { elementType: "labels.text.fill", stylers: [{ color: "#8d7458" }] },
        { featureType: "road", elementType: "geometry", stylers: [{ color: "#1a120d" }] },
        { featureType: "water", elementType: "geometry", stylers: [{ color: "#131416" }] },
      ],
    });

    state.mapInfoWindow = new window.google.maps.InfoWindow();
    setMapStatus("Map ready");
    return true;
  } catch (err) {
    setMapStatus("Map unavailable");
    addDebugLog(`map init failed: ${String(err.message || err)}`);
    return false;
  }
}

async function renderHotelMap(network) {
  const mapReady = await initHotelMap();
  if (!mapReady) {
    return;
  }

  clearMapMarkers();

  const hotels = (network.nodes || []).filter((node) => {
    const labels = node.labels || [];
    const lat = Number(node?.properties?.lat);
    const lng = Number(node?.properties?.lng);
    return labels.includes("Hotel") && Number.isFinite(lat) && Number.isFinite(lng);
  });

  if (!hotels.length) {
    setMapStatus("No hotel coordinates for current selection");
    return;
  }

  const bounds = new window.google.maps.LatLngBounds();
  for (const hotel of hotels) {
    const lat = Number(hotel.properties.lat);
    const lng = Number(hotel.properties.lng);
    const marker = new window.google.maps.Marker({
      map: state.map,
      position: { lat, lng },
      title: normalizeNodeName(hotel),
    });

    marker.addListener("click", () => {
      const price = hotel?.properties?.price_range || "-";
      const rating = hotel?.properties?.rating || "-";
      state.mapInfoWindow.setContent(
        `<div style="font-family:Space Grotesk,sans-serif;color:#2d2317;padding:4px 6px;min-width:170px;"><strong>${normalizeNodeName(hotel)}</strong><div>Rating: ${rating}</div><div>Price: ${price}</div></div>`
      );
      state.mapInfoWindow.open({ map: state.map, anchor: marker });
      loadNode(hotel.id);
    });

    state.mapMarkers.push(marker);
    bounds.extend({ lat, lng });
  }

  state.map.fitBounds(bounds, 60);
  if (hotels.length === 1) {
    state.map.setZoom(13);
  }
  setMapStatus(`Showing ${hotels.length} hotels`);
}

function askClearConfirmation() {
  return new Promise((resolve) => {
    els.confirmModal.classList.remove("hidden");
    els.confirmInput.value = "";
    els.confirmInput.focus();

    const cleanup = () => {
      els.confirmOkBtn.removeEventListener("click", onOk);
      els.confirmCancelBtn.removeEventListener("click", onCancel);
      els.confirmInput.removeEventListener("keydown", onKeyDown);
      els.confirmModal.classList.add("hidden");
    };

    const onOk = () => {
      const value = (els.confirmInput.value || "").trim();
      cleanup();
      resolve(value);
    };

    const onCancel = () => {
      cleanup();
      resolve("");
    };

    const onKeyDown = (evt) => {
      if (evt.key === "Enter") {
        evt.preventDefault();
        onOk();
      }
      if (evt.key === "Escape") {
        evt.preventDefault();
        onCancel();
      }
    };

    els.confirmOkBtn.addEventListener("click", onOk);
    els.confirmCancelBtn.addEventListener("click", onCancel);
    els.confirmInput.addEventListener("keydown", onKeyDown);
  });
}

async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `Request failed (${res.status})`);
  }
  return res.json();
}

function getCurrentCity() {
  const raw = (els.cityInput.value || "").trim();
  return raw.length > 0 ? raw : null;
}

function normalizeNodeName(node) {
  return (
    node?.properties?.name ||
    node?.properties?.title ||
    node?.properties?.id ||
    "Unknown"
  );
}

function topLabel(node) {
  return node.labels && node.labels.length ? node.labels[0] : "Node";
}

function renderOverview(overview, network, cityDisplay = null) {
  const city = cityDisplay || getCurrentCity() || "All";
  const cityStats = overview.city_stats || {};
  const stats = [
    { key: "Nodes", val: network.node_count || 0 },
    { key: "Edges", val: network.edge_count || 0 },
    { key: "City", val: city },
    {
      key: "Avg Rating",
      val: cityStats.avg_rating ? cityStats.avg_rating.toFixed(2) : "-",
    },
  ];

  els.statsGrid.innerHTML = stats
    .map(
      (item) =>
        `<div class="stat"><p>${item.key}</p><strong>${item.val}</strong></div>`
    )
    .join("");

  els.labelList.innerHTML = (overview.labels || [])
    .slice(0, 20)
    .map((item) => `<span class="tag">${item.label}: ${item.count}</span>`)
    .join("");

  els.relList.innerHTML = (overview.relationships || [])
    .slice(0, 20)
    .map((item) => `<span class="tag">${item.type}: ${item.count}</span>`)
    .join("");
}

function renderNodeTable(nodes) {
  const topNodes = [...nodes]
    .sort((a, b) => {
      const ar = Number(a.properties?.rating || 0);
      const br = Number(b.properties?.rating || 0);
      return br - ar;
    })
    .slice(0, 180);

  els.nodesTableBody.innerHTML = topNodes
    .map((node) => {
      const name = normalizeNodeName(node);
      const rating = node.properties?.rating ?? "-";
      const price = node.properties?.price_range ?? "-";
      return `
        <tr data-node-id="${node.id}">
          <td>${node.id.slice(-8)}</td>
          <td>${(node.labels || []).join(", ")}</td>
          <td>${name}</td>
          <td>${rating}</td>
          <td>${price}</td>
        </tr>
      `;
    })
    .join("");

  for (const row of els.nodesTableBody.querySelectorAll("tr")) {
    row.addEventListener("click", () => loadNode(row.dataset.nodeId));
  }
}

function graphElements(network) {
  const nodeEls = (network.nodes || []).map((node) => {
    const label = topLabel(node);
    return {
      data: {
        id: node.id,
        label,
        name: normalizeNodeName(node),
      },
      classes: `label-${label}`,
    };
  });

  const edgeEls = (network.edges || []).map((edge) => ({
    data: {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.type,
    },
  }));

  return [...nodeEls, ...edgeEls];
}

function legendLabels(nodes) {
  const found = new Set(nodes.map((n) => topLabel(n)));
  const labels = [...found].sort();
  els.graphLegend.innerHTML = labels
    .map((lbl) => {
      const color = labelPalette[lbl] || "#9ca9bf";
      return `<span class="tag"><span style="display:inline-block;width:9px;height:9px;background:${color};border-radius:50%;margin-right:6px;"></span>${lbl}</span>`;
    })
    .join("");
}

function renderGraph(network) {
  if (state.cy) {
    state.cy.destroy();
  }

  const cy = cytoscape({
    container: document.getElementById("cy"),
    elements: graphElements(network),
    layout: {
      name: "cose",
      animate: true,
      idealEdgeLength: 90,
      nodeRepulsion: 5500,
      animationDuration: 600,
    },
    style: [
      {
        selector: "node",
        style: {
          label: "data(name)",
          "font-size": 8,
          color: "#dce9ff",
          "text-wrap": "ellipsis",
          "text-max-width": 70,
          "background-color": "#7fa8ff",
          width: 16,
          height: 16,
          "border-width": 1,
          "border-color": "rgba(220,233,255,0.2)",
        },
      },
      {
        selector: "edge",
        style: {
          width: 1,
          "line-color": "rgba(148,169,206,0.4)",
          "target-arrow-color": "rgba(148,169,206,0.4)",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
        },
      },
      {
        selector: "node:selected",
        style: {
          width: 22,
          height: 22,
          "border-width": 2,
          "border-color": "#f5b700",
        },
      },
      ...Object.entries(labelPalette).map(([label, color]) => ({
        selector: `.label-${label}`,
        style: { "background-color": color },
      })),
    ],
  });

  cy.on("tap", "node", (evt) => {
    const nodeId = evt.target.id();
    loadNode(nodeId);
  });

  state.cy = cy;
  legendLabels(network.nodes || []);
}

function renderNodeDetails(detail) {
  const props = detail.properties || {};
  const labelText = (detail.labels || []).join(", ") || "Node";

  els.nodeMeta.innerHTML = [
    { k: "Node ID", v: detail.id || "-" },
    { k: "Labels", v: labelText },
    { k: "Neighbors", v: detail.total_neighbors || 0 },
  ]
    .map(
      (item) =>
        `<div class="kv-item"><p>${item.k}</p><strong>${item.v}</strong></div>`
    )
    .join("");

  els.nodeProps.textContent = JSON.stringify(props, null, 2);

  const neighbors = detail.neighbors || [];
  els.neighborList.innerHTML = neighbors.length
    ? neighbors
        .map((n) => {
          const other = n.other_node || {};
          const name =
            other.properties?.name || other.properties?.title || other.id || "Neighbor";
          return `
            <div class="neighbor-item">
              <div><strong>${n.relationship || "RELATED"}</strong> (${n.direction || "-"})</div>
              <div>${name}</div>
              <div class="muted">${(other.labels || []).join(", ")}</div>
            </div>
          `;
        })
        .join("")
    : '<p class="muted">No neighboring nodes found.</p>';
}

async function runQuery(evt) {
  evt.preventDefault();
  const question = (els.questionInput.value || "").trim();
  if (!question) {
    toast("Please enter a question", true);
    return;
  }

  els.answerText.textContent = "Generating answer...";
  els.contextText.textContent = "Loading context...";

  try {
    const payload = {
      question,
      city: getCurrentCity(),
    };
    const data = await request("/query", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    els.answerText.textContent = data.answer || "No answer returned.";
    els.contextText.textContent = data.context || "No context returned.";
    toast("Query completed");
  } catch (err) {
    els.answerText.textContent = "Failed to get answer.";
    els.contextText.textContent = String(err.message || err);
    toast("Query failed", true);
  }
}

async function loadNode(nodeId) {
  try {
    const data = await request(`/graph/node/${encodeURIComponent(nodeId)}?neighbor_limit=40`);
    renderNodeDetails(data);
  } catch (err) {
    toast("Unable to load node details", true);
  }
}

async function refreshDashboard() {
  const city = getCurrentCity();
  const limit = Number(els.limitInput.value || 180);

  els.refreshBtn.disabled = true;
  els.refreshBtn.textContent = "Loading...";

  try {
    const [overview, networkCity] = await Promise.all([
      request(`/graph/overview${city ? `?city=${encodeURIComponent(city)}` : ""}`),
      request("/graph/network", {
        method: "POST",
        body: JSON.stringify({ city, limit }),
      }),
    ]);

    let network = networkCity;
    let cityDisplay = city || "All";
    if (city && Number(networkCity.node_count || 0) === 0) {
      addDebugLog(`no nodes for city='${city}', loading full graph`);
      toast(`No graph data for '${city}'. Showing all cities.`);
      network = await request("/graph/network", {
        method: "POST",
        body: JSON.stringify({ city: null, limit }),
      });
      cityDisplay = `${city} (all cities)`;
    }

    state.overview = overview;
    state.network = network;

    renderOverview(overview, network, cityDisplay);
    renderNodeTable(network.nodes || []);
    renderGraph(network);
    renderHotelMap(network);

    if ((network.nodes || []).length > 0) {
      await loadNode(network.nodes[0].id);
    }

    toast("Graph refreshed");
  } catch (err) {
    setMapStatus("Map not updated");
    toast(`Refresh failed: ${String(err.message || err)}`, true);
  } finally {
    els.refreshBtn.disabled = false;
    els.refreshBtn.textContent = "Load Graph";
  }
}

async function pollIngestStatus(jobId) {
  try {
    const status = await request(`/ingest/status/${encodeURIComponent(jobId)}`);
    setProgress(status.progress || 0, status.step || status.status || "In progress", status);

    // Stream all new log lines we haven't shown yet
    if (Array.isArray(status.logs) && status.logs.length > state.seenLogCount) {
      const newLines = status.logs.slice(state.seenLogCount);
      for (const entry of newLines) {
        if (entry && entry.message) {
          addDebugLog(entry.message, entry.ts);
        }
      }
      state.seenLogCount = status.logs.length;
    }

    if (status.status === "completed") {
      if (state.ingestPolling) {
        window.clearInterval(state.ingestPolling);
        state.ingestPolling = null;
      }
      state.ingestJobId = null;
      state.seenLogCount = 0;
      els.applyBtn.disabled = false;
      els.applyBtn.textContent = "Start Ingest";
      toast("Ingestion complete. Loading graph...");
      addDebugLog(
        `completed: city=${status.result?.city || "-"}, hotels=${status.result?.hotels_ingested || 0}, news=${status.result?.news_ingested || 0}`
      );
      await refreshDashboard();
      return;
    }

    if (status.status === "failed") {
      if (state.ingestPolling) {
        window.clearInterval(state.ingestPolling);
        state.ingestPolling = null;
      }
      state.ingestJobId = null;
      state.seenLogCount = 0;
      els.applyBtn.disabled = false;
      els.applyBtn.textContent = "Start Ingest";
      setProgress(100, status.error || "Failed");
      toast("Ingestion failed", true);
      addDebugLog(`failed: ${status.error || "unknown error"}`);
    }
  } catch (err) {
    if (state.ingestPolling) {
      window.clearInterval(state.ingestPolling);
      state.ingestPolling = null;
    }
    state.ingestJobId = null;
    state.seenLogCount = 0;
    els.applyBtn.disabled = false;
    els.applyBtn.textContent = "Start Ingest";
    toast("Unable to check ingestion status", true);
    addDebugLog(`poll error: ${String(err.message || err)}`);
  }
}

async function applyAndIngest() {
  if (state.ingestJobId) {
    toast("Ingestion already running");
    return;
  }

  const city = getCurrentCity();
  if (!city) {
    toast("Enter a city first", true);
    return;
  }

  try {
    els.applyBtn.disabled = true;
    els.applyBtn.textContent = "Starting...";
    state.seenLogCount = 0;
    els.debugLog.textContent = "";
    setProgress(0, "Queued", { step_index: 0, step_total: 3, step_progress: 0 });
    addDebugLog(`starting ingest for city=${city}`);

    const start = await request("/ingest/start", {
      method: "POST",
      body: JSON.stringify({ city }),
    });

    state.ingestJobId = start.job_id;
    els.applyBtn.textContent = "Ingesting...";
    setProgress(start.progress || 0, start.step || "Queued", start);
    addDebugLog(`job created: ${start.job_id}`);

    state.ingestPolling = window.setInterval(() => {
      pollIngestStatus(start.job_id);
    }, 1300);

    await pollIngestStatus(start.job_id);
  } catch (err) {
    state.ingestJobId = null;
    els.applyBtn.disabled = false;
    els.applyBtn.textContent = "Start Ingest";
    setProgress(0, "Idle", { step_index: 0, step_total: 3, step_progress: 0 });
    toast("Failed to start ingestion", true);
    addDebugLog(`start error: ${String(err.message || err)}`);
  }
}

async function clearGraph() {
  const typed = await askClearConfirmation();
  if (!typed) {
    return;
  }

  if (typed.trim().toUpperCase() !== "DELETE ALL") {
    toast("Confirmation text mismatch", true);
    addDebugLog("clear graph cancelled: bad confirmation text");
    return;
  }

  try {
    els.clearBtn.disabled = true;
    els.clearBtn.textContent = "Clearing...";
    addDebugLog("clear graph requested");

    const result = await request("/graph/clear", {
      method: "POST",
      body: JSON.stringify({ confirm_text: typed }),
    });

    addDebugLog(
      `clear complete: nodes=${result.deleted_nodes || 0}, relationships=${result.deleted_relationships || 0}`
    );
    toast("Graph data cleared");

    renderOverview({ labels: [], relationships: [], city_stats: null }, { node_count: 0, edge_count: 0 }, "All");
    renderNodeTable([]);
    renderGraph({ nodes: [], edges: [] });
    renderHotelMap({ nodes: [], edges: [] });
    els.nodeMeta.innerHTML = "";
    els.nodeProps.textContent = "Graph is empty.";
    els.neighborList.innerHTML = '<p class="muted">No neighboring nodes found.</p>';
  } catch (err) {
    toast(`Clear failed: ${String(err.message || err)}`, true);
    addDebugLog(`clear graph failed: ${String(err.message || err)}`);
  } finally {
    els.clearBtn.disabled = false;
    els.clearBtn.textContent = "Clear Graph";
  }
}

function init() {
  els.cityInput.value = "Piliyandala";
  els.cityInput.addEventListener("keydown", (evt) => {
    if (evt.key === "Enter") {
      evt.preventDefault();
      applyAndIngest();
    }
  });
  els.queryForm.addEventListener("submit", runQuery);
  els.applyBtn.addEventListener("click", applyAndIngest);
  els.refreshBtn.addEventListener("click", refreshDashboard);
  els.clearBtn.addEventListener("click", clearGraph);
  setProgress(0, "Idle", { step_index: 0, step_total: 3, step_progress: 0 });
  addDebugLog("ui initialized");
  refreshDashboard();
}

init();
