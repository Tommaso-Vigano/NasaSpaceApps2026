
(() => {
    'use strict';
 
    // ------------------------------------------------------------------
    // Settings
    // ------------------------------------------------------------------
    const script = document.currentScript;
    const staticUrl = script.dataset.staticUrl;
    const CONFIG = {
        landmarksUrl: script.dataset.landmarksUrl,
        coverageUrl: script.dataset.coverageUrl,
        layersUrl: script.dataset.layersUrl,
        colorMapUrl: staticUrl + 'texture/Mars_surface_map.png',
        elevationMapUrl: staticUrl + 'texture/mars_elevation.jpg',
        starsUrl: staticUrl + 'texture/galaxy.svg',
 
        radius: 0.5,
 
        // Relief. Real Mars relief is only ~0.3% of its radius, so it is exaggerated.
        // Raise displacementScale for taller mountains, lower it for a smoother planet.
        displacementScale: 0.035,
        displacementBias: -0.015,   // shifts the whole surface so the mean sits near `radius`
        normalStrength: 0.6,        // how strongly the terrain is shaded (0 = flat look)
        reliefSmoothing: 2,         // blur radius (px) applied to the elevation map; higher = smoother
        segments: [1024, 512],      // planet mesh resolution (width, height); use [768, 384] on weak GPUs
        maxColorTextureWidth: 8192, // very large color maps are shrunk to this width to save GPU memory
 
        // Longitude (degrees East) found at the horizontal CENTER of each texture.
        // The bundled elevation map runs 0-360E from its left edge, so its center is 180.
        // If your color map is a -180..180 map (0 at the center), set colorMapCenterLon to 0.
        elevationMapCenterLon: 180,
        colorMapCenterLon: 180,
 
        rotationSpeed: 0.03,        // radians per second (about 3.5 minutes per turn)
    };
 
    const DEG = Math.PI / 180;
    const $ = (id) => document.getElementById(id);
 
    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    function loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error('Could not load ' + url));
            img.src = url;
        });
    }
 
    // ---- Relief processing ------------------------------------------------
    // The elevation map is small (1000 x 500), so using it directly shows its pixels as
    // blocky terraces. We upscale it 2x, blur it slightly, and build a smooth normal map from it.
    function blurHorizontal(src, dst, w, h, r) {
        const n = 2 * r + 1;
        for (let y = 0; y < h; y++) {
            const row = y * w;
            let sum = 0;
            for (let k = -r; k <= r; k++) sum += src[row + ((k + w) % w)];
            for (let x = 0; x < w; x++) {
                dst[row + x] = sum / n;
                sum += src[row + ((x + r + 1) % w)] - src[row + ((((x - r) % w) + w) % w)];
            }
        }
    }
 
    function blurVertical(src, dst, w, h, r) {
        const n = 2 * r + 1;
        for (let x = 0; x < w; x++) {
            let sum = 0;
            for (let k = -r; k <= r; k++) sum += src[Math.min(h - 1, Math.max(0, k)) * w + x];
            for (let y = 0; y < h; y++) {
                dst[y * w + x] = sum / n;
                sum += src[Math.min(h - 1, y + r + 1) * w + x] - src[Math.max(0, y - r) * w + x];
            }
        }
    }
 
    function buildRelief(img) {
        const w = img.naturalWidth * 2, h = img.naturalHeight * 2;
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, 0, 0, w, h);
        const rgba = ctx.getImageData(0, 0, w, h).data;
 
        let a = new Float32Array(w * h);
        let b = new Float32Array(w * h);
        for (let i = 0; i < a.length; i++) a[i] = rgba[i * 4] / 255;
 
        const r = Math.max(1, Math.round(CONFIG.reliefSmoothing));
        for (let pass = 0; pass < 2; pass++) {
            blurHorizontal(a, b, w, h, r);
            blurVertical(b, a, w, h, r);
        }
 
        // Smoothed height texture (used to displace the mesh)
        const heightCanvas = document.createElement('canvas');
        heightCanvas.width = w;
        heightCanvas.height = h;
        const hctx = heightCanvas.getContext('2d');
        const hImg = hctx.createImageData(w, h);
        for (let i = 0; i < a.length; i++) {
            const g = Math.round(a[i] * 255);
            hImg.data[i * 4] = g;
            hImg.data[i * 4 + 1] = g;
            hImg.data[i * 4 + 2] = g;
            hImg.data[i * 4 + 3] = 255;
        }
        hctx.putImageData(hImg, 0, 0);
 
        // Normal map (used for shading). Slope is scaled to match the exaggerated displacement.
        const pixelWorldSize = (2 * Math.PI * CONFIG.radius) / w;
        const strength = CONFIG.normalStrength * CONFIG.displacementScale / pixelWorldSize;
        const normalCanvas = document.createElement('canvas');
        normalCanvas.width = w;
        normalCanvas.height = h;
        const nctx = normalCanvas.getContext('2d');
        const nImg = nctx.createImageData(w, h);
        for (let y = 0; y < h; y++) {
            const ym = Math.max(0, y - 1), yp = Math.min(h - 1, y + 1);
            for (let x = 0; x < w; x++) {
                const xm = (x - 1 + w) % w, xp = (x + 1) % w;
                const dx = (a[y * w + xp] - a[y * w + xm]) * 0.5 * strength;
                const dy = (a[yp * w + x] - a[ym * w + x]) * 0.5 * strength;
                const inv = 1 / Math.sqrt(dx * dx + dy * dy + 1);
                const o = (y * w + x) * 4;
                nImg.data[o] = (-dx * inv * 0.5 + 0.5) * 255;
                nImg.data[o + 1] = (dy * inv * 0.5 + 0.5) * 255;
                nImg.data[o + 2] = (inv * 0.5 + 0.5) * 255;
                nImg.data[o + 3] = 255;
            }
        }
        nctx.putImageData(nImg, 0, 0);
 
        return { w, h, heights: a, heightCanvas, normalCanvas };
    }
 
    // Looks up the smoothed elevation (0..1, bilinear) so pins can sit on top of the terrain.
    function createElevationSampler(relief) {
        const { w, h, heights } = relief;
        const at = (x, y) => {
            const xx = ((x % w) + w) % w;
            const yy = Math.min(h - 1, Math.max(0, y));
            return heights[yy * w + xx];
        };
        return (lat, lon) => {
            const u = (((0.5 + (lon - CONFIG.elevationMapCenterLon) / 360) % 1) + 1) % 1;
            const v = (lat + 90) / 180;
            const fx = u * w - 0.5, fy = (1 - v) * h - 0.5;
            const x0 = Math.floor(fx), y0 = Math.floor(fy);
            const tx = fx - x0, ty = fy - y0;
            const top = at(x0, y0) * (1 - tx) + at(x0 + 1, y0) * tx;
            const bottom = at(x0, y0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1) * tx;
            return top * (1 - ty) + bottom * ty;
        };
    }
 
    // Latitude / East longitude -> position on a sphere, matching THREE.SphereGeometry's UV layout.
    function latLonToVector3(lat, lon, r) {
        const phi = lat * DEG;
        const lam = (lon - CONFIG.elevationMapCenterLon) * DEG;
        return new THREE.Vector3(
            r * Math.cos(phi) * Math.cos(lam),
            r * Math.sin(phi),
            -r * Math.cos(phi) * Math.sin(lam)
        );
    }
 
    function formatCoords(lat, lon) {
        const ns = lat >= 0 ? 'N' : 'S';
        return Math.abs(lat).toFixed(2) + '\u00b0' + ns + ', ' + (((lon % 360) + 360) % 360).toFixed(2) + '\u00b0E';
    }
 
    function formatDate(iso) {
        return new Date(iso + 'T00:00:00Z').toLocaleDateString('en-GB', {
            day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC',
        });
    }
 
    // Tiny DOM builder: el('div', { class: 'x' }, child, child...)
    function el(tag, attrs, ...children) {
        const node = document.createElement(tag);
        Object.entries(attrs || {}).forEach(([key, value]) => {
            if (value === false || value == null) return;
            if (key === 'class') node.className = value;
            else node.setAttribute(key, value === true ? '' : value);
        });
        children.forEach((c) => node.append(c));
        return node;
    }
 
    const ROVER_SVG =
        '<svg viewBox="0 0 18 12" aria-hidden="true" fill="currentColor">' +
        '<rect x="3" y="4" width="10" height="3" rx="0.8"/><rect x="11" y="0.8" width="1" height="3.4"/>' +
        '<rect x="10.4" y="0" width="3" height="1.6" rx="0.4"/>' +
        '<circle cx="4" cy="9.4" r="2"/><circle cx="8" cy="9.4" r="2"/><circle cx="12" cy="9.4" r="2"/></svg>';
 
    function createSwatch(type) {
        const swatch = el('span', { class: 'swatch' + (type.model === 'rover' ? '' : ' swatch-dot') });
        swatch.style.setProperty('--c', type.color);
        if (type.model === 'rover') swatch.innerHTML = ROVER_SVG;
        return swatch;
    }
 
    // The star background is an SVG. Rasterize it once at a fixed size so it stays sharp
    // and behaves like any other texture on the inside of the sky sphere.
    async function loadStarTexture(url, renderer) {
        const img = await loadImage(url);
        const width = Math.min(4096, renderer.capabilities.maxTextureSize);
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = width / 2;
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        const texture = new THREE.CanvasTexture(canvas);
        texture.encoding = THREE.sRGBEncoding;
        texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
        return texture;
    }
 
    // Color maps are usually centered differently from the elevation map, and a very large
    // one (tens of MB) can exceed what a GPU holds comfortably. Both are handled by drawing the
    // image into a canvas, shifted sideways (wrapping around) and/or scaled down.
    async function loadColorTexture(url, renderer, centerLongitude = CONFIG.colorMapCenterLon) {
        const img = await loadImage(url);
        const shift = (CONFIG.elevationMapCenterLon - centerLongitude) / 360;
        const maxWidth = Math.min(CONFIG.maxColorTextureWidth, renderer.capabilities.maxTextureSize);
        const scale = Math.min(1, maxWidth / img.naturalWidth);
        let texture;
 
        if (Math.abs(shift) < 1e-6 && scale === 1) {
            texture = new THREE.Texture(img);
        } else {
            const cw = Math.round(img.naturalWidth * scale);
            const ch = Math.round(img.naturalHeight * scale);
            const canvas = document.createElement('canvas');
            canvas.width = cw;
            canvas.height = ch;
            const ctx = canvas.getContext('2d');
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            const dx = -shift * cw;
            [-1, 0, 1].forEach((k) => ctx.drawImage(img, dx + k * cw, 0, cw, ch));
            texture = new THREE.CanvasTexture(canvas);
        }
 
        texture.encoding = THREE.sRGBEncoding;
        texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
        texture.needsUpdate = true;
        return texture;
    }
 
    // ------------------------------------------------------------------
    // Main
    // ------------------------------------------------------------------
    async function init() {
        const canvas = $('globe');
        const labelEl = $('label');
        const labelName = $('label-name');
        const labelMeta = $('label-meta');
        const labelDetail = $('label-detail');
        const labelCoords = $('label-coords');
        const toggleBtn = $('rotate-toggle');
        const loadingEl = $('loading');
 
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.outputEncoding = THREE.sRGBEncoding;
 
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 100);
        camera.position.set(0, 0.25, 1.6);
        scene.add(camera);
 
        // Load the data in parallel
        const [elevationImg, colorMap, starTexture, landmarkData, coverageData] = await Promise.all([
            loadImage(CONFIG.elevationMapUrl),
            loadColorTexture(CONFIG.colorMapUrl, renderer),
            loadStarTexture(CONFIG.starsUrl, renderer),
            fetch(CONFIG.landmarksUrl).then((r) => {
                if (!r.ok) throw new Error('Could not load ' + CONFIG.landmarksUrl);
                return r.json();
            }),
            fetch(CONFIG.coverageUrl).then((response) => {
                if (!response.ok) throw new Error('Could not load elevation coverage');
                return response.json();
            }).catch((error) => {
                console.warn(error);
                return null; // Coverage is optional; the globe remains usable.
            }),
        ]);
 
        const relief = buildRelief(elevationImg);
        const sampleElevation = createElevationSampler(relief);
 
        // ---- Planet with real relief --------------------------------
        const elevationMap = new THREE.CanvasTexture(relief.heightCanvas);
        elevationMap.minFilter = THREE.LinearFilter;
        elevationMap.magFilter = THREE.LinearFilter;
        elevationMap.generateMipmaps = false;
        elevationMap.needsUpdate = true;
 
        const normalMap = new THREE.CanvasTexture(relief.normalCanvas);
        normalMap.anisotropy = renderer.capabilities.getMaxAnisotropy();
        normalMap.needsUpdate = true;
 
        // Many segments, because displacement moves actual vertices
        const geometry = new THREE.SphereGeometry(CONFIG.radius, CONFIG.segments[0], CONFIG.segments[1]);
        const material = new THREE.MeshPhongMaterial({
            map: colorMap,
            normalMap: normalMap,
            displacementMap: elevationMap,
            displacementScale: CONFIG.displacementScale,
            displacementBias: CONFIG.displacementBias,
            specular: 0x050505,
            shininess: 2,
        });
        // Tint the existing displaced surface rather than placing translucent
        // patch meshes over it. A single "inside any rectangle" test renders
        // the union exactly, without darker overlaps, seams or z-fighting.
        const coverageToggle = $('coverage-visible');
        const coverageStatus = $('coverage-status');
        const coverageEnabled = { value: coverageToggle.checked ? 1.0 : 0.0 };
        if (coverageData && Array.isArray(coverageData.rectangles)) {
            const rectangles = coverageData.rectangles.map((rectangle) => new THREE.Vector4(
                rectangle.top_left[0], rectangle.bottom_right[1],
                rectangle.bottom_right[0], rectangle.top_left[1],
            )); // west, south, east, north; all angles in degrees
            coverageStatus.textContent = rectangles.length
                ? 'Mint arrows point to elevation coverage. Gaps within datasets may exist.'
                : 'No elevation datasets loaded.';
            coverageToggle.disabled = rectangles.length === 0;
            if (rectangles.length) {
                material.onBeforeCompile = (shader) => {
                    shader.uniforms.coverageEnabled = coverageEnabled;
                    shader.uniforms.coverageRectangles = { value: rectangles };
                    shader.vertexShader = 'varying vec3 vCoveragePosition;\n' + shader.vertexShader;
                    shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>',
                        '#include <begin_vertex>\nvCoveragePosition = position;');
                    shader.fragmentShader = `
                        varying vec3 vCoveragePosition;
                        uniform float coverageEnabled;
                        uniform vec4 coverageRectangles[${rectangles.length}];
                        float elevationCoverage(vec2 coordinate) {
                            float covered = 0.0;
                            for (int i = 0; i < ${rectangles.length}; i++) {
                                vec4 box = coverageRectangles[i];
                                // Accept both signed and 0..360 source longitudes.
                                float lon = coordinate.x;
                                bool withinLongitude =
                                    (lon >= box.x && lon <= box.z) ||
                                    (lon + 360.0 >= box.x && lon + 360.0 <= box.z) ||
                                    (lon - 360.0 >= box.x && lon - 360.0 <= box.z);
                                if (withinLongitude && coordinate.y >= box.y && coordinate.y <= box.w)
                                    covered = 1.0;
                            }
                            return covered;
                        }
                    ` + shader.fragmentShader;
                    shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', `
                        #include <color_fragment>
                        vec3 direction = normalize(vCoveragePosition);
                        vec2 coordinate = vec2(
                            mod(degrees(atan(-direction.z, direction.x)) + ${CONFIG.elevationMapCenterLon.toFixed(8)} + 540.0, 360.0) - 180.0,
                            degrees(asin(clamp(direction.y, -1.0, 1.0)))
                        );
                        float covered = elevationCoverage(coordinate) * coverageEnabled;
                        diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.16, 0.85, 0.61), covered * 0.60);
                    `);
                };
            }
        } else {
            coverageStatus.textContent = 'Elevation coverage unavailable. Reload to retry.';
            coverageToggle.disabled = true;
        }
        coverageToggle.addEventListener('change', () => {
            coverageEnabled.value = coverageToggle.checked ? 1.0 : 0.0;
            coverageToggle.closest('label').classList.toggle('is-off', !coverageToggle.checked);
            coverageArrows.visible = coverageToggle.checked;
        });

        const planetMesh = new THREE.Mesh(geometry, material);
 
        // Everything that should turn with the planet (surface + pins) lives in this group
        const planet = new THREE.Group();
        planet.add(planetMesh);
        scene.add(planet);
 
        // Replace only the colour map. Relief, coverage, pins and selection
        // remain in the same geographic frame for every visualization mode.
        async function setupVisualizationLayers() {
            const options = $('layer-options');
            const status = $('layer-status');
            const legend = $('layer-legend');
            const textures = new Map();
            let requestVersion = 0;
            let activeLayer = 'surface';
            function applyLayer(id, texture, metadata) {
                material.map = texture;
                material.needsUpdate = true;
                activeLayer = id;
                options.value = id;
                legend.hidden = !metadata;
                const isImage = metadata && metadata.legend_type === 'image';
                $('layer-scale').hidden = !metadata || isImage;
                $('layer-legend-title').textContent = metadata ? metadata.label : '';
                $('layer-units').textContent = isImage
                    ? 'Pre-colored imagery. Numeric color scale unavailable in this file.'
                    : metadata ? (metadata.units || 'Dimensionless') : '';
                status.hidden = true;
                status.textContent = '';
                if (metadata && metadata.legend_type !== 'image') {
                    const number = (value) => new Intl.NumberFormat(undefined,
                        {maximumSignificantDigits: 4}).format(value);
                    $('layer-gradient').style.background =
                        'linear-gradient(to top, ' + metadata.colors.join(', ') + ')';
                    $('layer-min').textContent = number(metadata.minimum);
                    $('layer-max').textContent = number(metadata.maximum);
                    legend.title = metadata.stretch + '; dark grey: missing';
                }
                if (isImage || !metadata) legend.removeAttribute('title');
            }
            options.addEventListener('change', () => {
                if (options.value !== 'surface') return;
                requestVersion++;
                options.removeAttribute('aria-busy');
                applyLayer('surface', colorMap, null);
            });
            try {
                const response = await fetch(CONFIG.layersUrl);
                if (!response.ok) throw new Error('Could not load visualization modes');
                const catalog = await response.json();
                for (const layer of catalog.layers) {
                    const option = document.createElement('option');
                    option.value = layer.id;
                    option.textContent = layer.label;
                    option.disabled = !layer.available;
                    if (!layer.available) option.title = 'This dataset is not currently available';
                    options.append(option);
                }
                options.addEventListener('change', async () => {
                        const layer = catalog.layers.find(layer => layer.id === options.value);
                        if (!layer || !layer.available) return;
                        const version = ++requestVersion;
                        status.hidden = false;
                        status.textContent = 'Loading ' + layer.label.toLowerCase() + '...';
                        options.setAttribute('aria-busy', 'true');
                        try {
                            if (!textures.has(layer.id)) {
                                const pending = (async () => {
                                    const response = await fetch(layer.metadata_url);
                                    if (!response.ok) {
                                        const error = await response.json();
                                        throw new Error(error.error || 'Layer could not be loaded');
                                    }
                                    const metadata = await response.json();
                                    const texture = await loadColorTexture(metadata.texture_url, renderer,
                                                                           metadata.center_longitude);
                                    return {texture, metadata};
                                })();
                                textures.set(layer.id, pending);
                                pending.catch(() => textures.delete(layer.id));
                            }
                            const {texture, metadata} = await textures.get(layer.id);
                            // A slower, earlier request must never override a newer selection.
                            if (version === requestVersion) applyLayer(layer.id, texture, metadata);
                        } catch (error) {
                            console.warn(error);
                            if (version === requestVersion) {
                                options.value = activeLayer;
                                status.textContent = 'Could not load ' + layer.label.toLowerCase()
                                    + '. Select it again to retry.';
                            }
                        } finally {
                            if (version === requestVersion) options.removeAttribute('aria-busy');
                        }
                });
                if (!catalog.layers.some((layer) => layer.available)) {
                    status.hidden = false;
                    status.textContent = 'Surface imagery | additional datasets unavailable';
                }
            } catch (error) {
                console.warn(error);
                status.hidden = false;
                status.textContent = 'Surface imagery | additional modes unavailable';
            }
        }
        setupVisualizationLayers();

        // Four inward-pointing surface arrows make small coverage patches
        // discoverable from orbit. Their size is angular, independent of the
        // patch size; the tip stops just short of the actual footprint.
        const coverageArrows = new THREE.Group();
        coverageArrows.visible = coverageToggle.checked;
        planet.add(coverageArrows);
        const coverageArrowMaterial = new THREE.MeshBasicMaterial({
            color: 0x98fb98, side: THREE.DoubleSide, depthWrite: false,
            transparent: true, opacity: 0.95,
        });
        const coverageArrowAccent = new THREE.MeshBasicMaterial({
            color: 0xc8ffe2, side: THREE.DoubleSide, depthWrite: false,
            transparent: true, opacity: 0.55,
        });
        const coverageArrowGlow = new THREE.MeshBasicMaterial({
            color: 0x98fb98, side: THREE.DoubleSide, depthWrite: false,
            transparent: true, opacity: 0.12, blending: THREE.AdditiveBlending,
        });
        function addCoverageArrow(latitude, longitude, axis, sign) {
            const phi = latitude * DEG;
            const lambda = (longitude - CONFIG.elevationMapCenterLon) * DEG;
            const normal = latLonToVector3(latitude, longitude, 1);
            const north = new THREE.Vector3(-Math.sin(phi) * Math.cos(lambda),
                Math.cos(phi), Math.sin(phi) * Math.sin(lambda));
            const east = new THREE.Vector3(-Math.sin(lambda), 0, -Math.cos(lambda));
            const outward = (axis === 'latitude' ? north : east).clone().multiplyScalar(sign);
            const sideways = new THREE.Vector3().crossVectors(normal, outward).normalize();
            function point(distance, lateral) {
                // Spherical tangent offsets also behave correctly at the poles
                // and across the longitude seam.
                const tangent = outward.clone().multiplyScalar(distance * DEG)
                    .addScaledVector(sideways, lateral * DEG);
                const angle = tangent.length();
                const direction = normal.clone().multiplyScalar(Math.cos(angle))
                    .addScaledVector(tangent.normalize(), Math.sin(angle)).normalize();
                const lat = Math.asin(THREE.MathUtils.clamp(direction.y, -1, 1)) / DEG;
                const lon = Math.atan2(-direction.z, direction.x) / DEG + CONFIG.elevationMapCenterLon;
                const radius = CONFIG.radius + sampleElevation(lat, lon) * CONFIG.displacementScale
                    + CONFIG.displacementBias + 0.002;
                return direction.multiplyScalar(radius);
            }
            function chevron(tip, back, width, thickness, mat) {
                const vertices = [], indices = [];
                // Subdivide each solid arm so it hugs the displaced surface.
                for (const side of [-1, 1]) {
                    const offset = vertices.length / 3;
                    for (let i = 0; i <= 12; i++) {
                        const t = i / 12;
                        const outside = point(tip + (back - tip) * t, side * width * t);
                        const inside = point(tip + thickness + (back - tip - thickness) * t,
                                             side * (width - thickness * 0.65) * t);
                        vertices.push(outside.x, outside.y, outside.z, inside.x, inside.y, inside.z);
                        if (i < 12) {
                            const n = offset + i * 2;
                            indices.push(n, n + 1, n + 2, n + 1, n + 3, n + 2);
                        }
                    }
                }
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
                geometry.setIndex(indices);
                coverageArrows.add(new THREE.Mesh(geometry, mat));
            }
            // A broad, shaftless head with a detached trailing accent: compact
            // navigation-glyph proportions rather than a long diagram arrow.
            chevron(0.85, 2.85, 1.55, 1.0, coverageArrowGlow);
            chevron(1.0, 2.7, 1.4, 0.65, coverageArrowMaterial);
            chevron(2.5, 3.65, 1.15, 0.32, coverageArrowAccent);
        }
        const arrowFootprints = new Set();
        for (const rectangle of (coverageData && coverageData.rectangles) || []) {
            const key = JSON.stringify(rectangle);
            if (arrowFootprints.has(key)) continue;
            arrowFootprints.add(key);
            const [west, north] = rectangle.top_left;
            const [east, south] = rectangle.bottom_right;
            if (east - west >= 359 && north - south >= 179) continue;
            const middleLatitude = (north + south) / 2;
            const middleLongitude = (west + east) / 2;
            addCoverageArrow(north, middleLongitude, 'latitude', 1);
            addCoverageArrow(south, middleLongitude, 'latitude', -1);
            addCoverageArrow(middleLatitude, east, 'longitude', 1);
            addCoverageArrow(middleLatitude, west, 'longitude', -1);
        }

        // ---- Stars ---------------------------------------------------
        scene.add(new THREE.Mesh(
            new THREE.SphereGeometry(50, 48, 48),
            new THREE.MeshBasicMaterial({ map: starTexture, side: THREE.BackSide })
        ));
 
        // ---- Light (follows the camera so the visible side is always lit) ----
        scene.add(new THREE.AmbientLight(0xffffff, 0.2));
        const sun = new THREE.DirectionalLight(0xffffff, 1.1);
        sun.position.set(-3, 1.5, 2);
        camera.add(sun);
 
        // ---- Landmark types -----------------------------------------
        const types = (landmarkData.types && landmarkData.types.length)
            ? landmarkData.types
            : [{ id: 'default', group: 'features', label: 'Landmarks', singular: 'Landmark', color: '#f0c9a4' }];
        const typeById = {};
        types.forEach((t) => { typeById[t.id] = t; t.singular = t.singular || t.label; });
 
        const groups = (landmarkData.groups || []).slice();
        types.forEach((t) => {
            if (!groups.some((g) => g.id === t.group)) groups.push({ id: t.group || 'other', label: 'Other' });
            t.group = t.group || 'other';
        });
 
        const typeVisible = {};
        types.forEach((t) => { typeVisible[t.id] = true; });
 
        // ---- Landmark pins ------------------------------------------
        // Every pin is a fine needle with a soft head and a thin ground ring, tinted with its
        // type's pastel color. Rover sites get a tiny rover on the needle instead of a head.
        const WHITE = new THREE.Color(0xffffff);
        const tint = (color, amount) => color.clone().lerp(WHITE, amount);
 
        const STEM_LENGTH = 0.024;
        const ROVER_SCALE = 2.0;
        const markers = [];
        const UP = new THREE.Vector3(0, 1, 0);
 
        const stemGeo = new THREE.CylinderGeometry(0.0006, 0.0006, STEM_LENGTH, 5);
        stemGeo.translate(0, STEM_LENGTH / 2, 0);
        const headGeo = new THREE.SphereGeometry(0.0045, 14, 14);
        const haloGeo = new THREE.SphereGeometry(0.0095, 14, 14);
        const ringGeo = new THREE.RingGeometry(0.0062, 0.0074, 48);
        ringGeo.rotateX(-Math.PI / 2);                      // lie flat on the ground
        const hitGeo = new THREE.SphereGeometry(0.02, 8, 8);
 
        // Shared low-poly geometry keeps all rover pins lightweight.
        const roverGeo = {
            body: new THREE.BoxGeometry(0.012, 0.0032, 0.0078),
            deck: new THREE.BoxGeometry(0.014, 0.0008, 0.009),
            wheel: new THREE.CylinderGeometry(0.0025, 0.0025, 0.0018, 12).rotateX(Math.PI / 2),
            hub: new THREE.CylinderGeometry(0.0010, 0.0010, 0.0019, 10).rotateX(Math.PI / 2),
            strut: new THREE.CylinderGeometry(0.00035, 0.00035, 1, 6),
            mast: new THREE.CylinderGeometry(0.0005, 0.0006, 0.0075, 6),
            camera: new THREE.BoxGeometry(0.0022, 0.0021, 0.0054),
            lens: new THREE.CylinderGeometry(0.00065, 0.00065, 0.0002, 10).rotateZ(Math.PI / 2),
            dish: new THREE.CylinderGeometry(0.0023, 0.0016, 0.0006, 16),
        };

        function makeSphereHead(color) {
            const baseColor = tint(color, 0.12);
            const mat = new THREE.MeshBasicMaterial({ color: baseColor });
            const head = new THREE.Mesh(headGeo, mat);
            head.position.y = STEM_LENGTH;
            const haloMat = new THREE.MeshBasicMaterial({
                color, transparent: true, opacity: 0.16, depthWrite: false,
            });
            head.add(new THREE.Mesh(haloGeo, haloMat));
            return {
                object: head,
                setSelected(on) {
                    mat.color.copy(on ? tint(color, 0.5) : baseColor);
                    haloMat.opacity = on ? 0.32 : 0.16;
                    head.scale.setScalar(on ? 1.5 : 1);
                },
            };
        }
 
        function makeRoverHead(color) {
            const root = new THREE.Group();
            root.scale.setScalar(ROVER_SCALE);
            root.position.y = STEM_LENGTH + 0.0025 * ROVER_SCALE;
            root.rotation.y = -Math.PI / 8; // A gentle three-quarter silhouette.
 
            const parts = [];
            const material = (tone, glow) => {
                const m = new THREE.MeshLambertMaterial({
                    color: color.clone().multiplyScalar(tone),
                    emissive: color.clone().multiplyScalar(glow),
                });
                parts.push({ m, tone, glow });
                return m;
            };
            const bodyMat = material(1, 0.35);
            const wheelMat = material(0.25, 0.05);
            const trimMat = material(0.8, 0.25);
            const mastMat = material(1.1, 0.4);
            const lensMat = material(0.12, 0.02);

            function part(geometry, mat, x, y, z) {
                const mesh = new THREE.Mesh(geometry, mat);
                mesh.position.set(x, y, z);
                root.add(mesh);
                return mesh;
            }

            function strut(from, to) {
                const start = new THREE.Vector3(...from);
                const end = new THREE.Vector3(...to);
                const beam = part(roverGeo.strut, trimMat, 0, 0, 0);
                beam.position.copy(start).add(end).multiplyScalar(0.5);
                beam.quaternion.setFromUnitVectors(UP, end.clone().sub(start).normalize());
                beam.scale.y = start.distanceTo(end);
            }

            part(roverGeo.body, bodyMat, 0, 0.0035, 0);
            part(roverGeo.deck, mastMat, 0, 0.0055, 0);

            // Six distinct dark tyres, pale hubs and an articulated suspension.
            [-1, 1].forEach((side) => {
                const z = side * 0.0062;
                [-0.0056, 0, 0.0056].forEach((x) => {
                    part(roverGeo.wheel, wheelMat, x, 0, z);
                    part(roverGeo.hub, trimMat, x, 0, z + side * 0.00015);
                    strut([x, 0, z], [x * 0.55, 0.0035, side * 0.0035]);
                });
                strut([-0.0056, 0.001, z], [0, 0.0027, z]);
                strut([0, 0.0027, z], [0.0056, 0.001, z]);
            });

            // Tall binocular camera mast is readable even at globe scale.
            part(roverGeo.mast, mastMat, 0.004, 0.0096, 0);
            part(roverGeo.camera, mastMat, 0.0043, 0.0136, 0);
            [-0.0014, 0.0014].forEach((z) => {
                part(roverGeo.lens, lensMat, 0.0055, 0.0136, z);
            });
            const dish = part(roverGeo.dish, trimMat, -0.0035, 0.008, 0);
            dish.rotation.z = -Math.PI / 7;
            strut([-0.0035, 0.0057, 0], [-0.0035, 0.008, 0]);

            const haloMat = new THREE.MeshBasicMaterial({
                color, transparent: true, opacity: 0.16, depthWrite: false,
            });
            const halo = new THREE.Mesh(haloGeo, haloMat);
            halo.scale.setScalar(1.0);
            halo.position.y = 0.005;
            root.add(halo);
 
            return {
                object: root,
                setSelected(on) {
                    parts.forEach(({ m, tone, glow }) => {
                        m.color.copy(color).multiplyScalar(tone);
                        if (on) m.color.lerp(WHITE, 0.3);
                        m.emissive.copy(color).multiplyScalar(on ? glow + 0.3 : glow);
                    });
                    haloMat.opacity = on ? 0.23 : 0.10;
                    root.scale.setScalar(on ? ROVER_SCALE * 1.15 : ROVER_SCALE);
                },
            };
        }
 
        (landmarkData.landmarks || []).forEach((lm) => {
            const type = typeById[lm.type] || types[0];
            const color = new THREE.Color(type.color);
 
            const height = sampleElevation(lm.latitude, lm.longitude);
            const r = CONFIG.radius + height * CONFIG.displacementScale + CONFIG.displacementBias + 0.002;
            const position = latLonToVector3(lm.latitude, lm.longitude, r);
 
            const group = new THREE.Group();
            group.position.copy(position);
            group.quaternion.setFromUnitVectors(UP, position.clone().normalize());
 
            const stemMat = new THREE.MeshBasicMaterial({
                color: tint(color, 0.08), transparent: true, opacity: 0.85,
            });
            const stem = new THREE.Mesh(stemGeo, stemMat);
 
            const headApi = type.model === 'rover' ? makeRoverHead(color) : makeSphereHead(color);
 
            const ringMat = new THREE.MeshBasicMaterial({
                color, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthWrite: false,
            });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.position.y = 0.002;
 
            // Larger invisible sphere so the delicate pins are still easy to click
            const hit = new THREE.Mesh(hitGeo, new THREE.MeshBasicMaterial({
                transparent: true, opacity: 0, depthWrite: false,
            }));
            hit.position.y = type.model === 'rover'
                ? STEM_LENGTH + 0.006 * ROVER_SCALE : STEM_LENGTH * 0.7;
            if (type.model === 'rover') hit.scale.setScalar(1.25);
 
            group.add(stem, headApi.object, ring, hit);
            planet.add(group);
 
            const marker = {
                landmark: lm, type, group, hit,
                head: headApi.object,
                vis: 1, target: 1,        // current / wanted visibility (0..1) for the grow-in animation
                setSelected(on) {
                    headApi.setSelected(on);
                    ringMat.color.copy(on ? tint(color, 0.4) : color);
                    ring.scale.setScalar(on ? 1.6 : 1);
                },
            };
            hit.userData.marker = marker;
            markers.push(marker);
        });
 
        // ---- Controls ------------------------------------------------
        const controls = new THREE.OrbitControls(camera, canvas);
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;
        controls.enablePan = false;
        controls.minDistance = 0.7;
        controls.maxDistance = 3.5;
        controls.zoomSpeed = 0.7;
 
        // ---- Auto-rotation toggle ------------------------------------
        let rotating = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        let rotationSpeed = rotating ? CONFIG.rotationSpeed : 0;
 
        function renderToggle() {
            toggleBtn.setAttribute('aria-pressed', String(!rotating));
            toggleBtn.querySelector('.toggle-text').textContent = rotating ? 'Pause rotation' : 'Resume rotation';
        }
        toggleBtn.addEventListener('click', () => {
            rotating = !rotating;
            renderToggle();
        });
        renderToggle();
 
        // ---- Picking / selection -------------------------------------
        const raycaster = new THREE.Raycaster();
        const ndc = new THREE.Vector2();
        const occluder = new THREE.Sphere(new THREE.Vector3(), CONFIG.radius - 0.01);
        const occRay = new THREE.Ray();
        const tmpA = new THREE.Vector3();
        const tmpB = new THREE.Vector3();
        let selected = null;
 
        // True when the planet's body is between the camera and this point
        function isOccluded(worldPos) {
            tmpA.copy(worldPos).sub(camera.position);
            const dist = tmpA.length();
            occRay.set(camera.position, tmpA.normalize());
            const entry = occRay.intersectSphere(occluder, tmpB);
            return entry !== null && camera.position.distanceTo(entry) < dist - 0.002;
        }
 
        function findMarkerAt(clientX, clientY) {
            const rect = canvas.getBoundingClientRect();
            ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
            raycaster.setFromCamera(ndc, camera);
            const pickable = markers.filter((m) => m.target === 1 && m.vis > 0.6).map((m) => m.hit);
            const hits = raycaster.intersectObjects(pickable, false);
            for (const hit of hits) {
                const marker = hit.object.userData.marker;
                if (!isOccluded(marker.head.getWorldPosition(tmpA.clone()))) return marker;
            }
            return null;
        }
 
        function select(marker) {
            if (selected === marker) return;
            if (selected) selected.setSelected(false);
            selected = marker;
            if (marker) {
                marker.setSelected(true);
                const lm = marker.landmark;
                labelName.textContent = lm.name;
                labelMeta.textContent = marker.type.singular + (lm.place ? ' in ' + lm.place : '');
                labelDetail.textContent = lm.landed ? 'Landed ' + formatDate(lm.landed) : '';
                labelDetail.hidden = !lm.landed;
                labelCoords.textContent = formatCoords(marker.landmark.latitude, marker.landmark.longitude);
                labelEl.hidden = false;
                updateLabel();
            } else {
                labelEl.hidden = true;
            }
        }
 
        const labelWorld = new THREE.Vector3();
        function updateLabel() {
            if (!selected) return;
            selected.head.getWorldPosition(labelWorld);
            const behind = isOccluded(labelWorld);
            labelEl.classList.toggle('is-behind', behind);
            labelWorld.project(camera);
            const x = (labelWorld.x * 0.5 + 0.5) * window.innerWidth;
            const y = (-labelWorld.y * 0.5 + 0.5) * window.innerHeight;
            labelEl.style.transform =
                'translate(' + x.toFixed(1) + 'px,' + y.toFixed(1) + 'px) translate(-50%, calc(-100% - 16px))';
        }
 
        // ---- Geographic region selection ---------------------------------
        // Intersect the reference sphere, then undo the planet's rotation.
        // This matches the longitude convention used by latLonToVector3.
        const selectionSphere = new THREE.Sphere(new THREE.Vector3(), CONFIG.radius);
        const roiLayer = new THREE.Group();
        planet.add(roiLayer);
        let roiDrag = null;
        let region = null;
        const wrapLongitude = (lon) => ((lon + 180) % 360 + 360) % 360 - 180;

        function surfaceCoordinate(e) {
            const rect = canvas.getBoundingClientRect();
            ndc.set((e.clientX - rect.left) / rect.width * 2 - 1,
                    1 - (e.clientY - rect.top) / rect.height * 2);
            camera.updateMatrixWorld();
            planet.updateMatrixWorld(true);
            raycaster.setFromCamera(ndc, camera);
            const hit = raycaster.ray.intersectSphere(selectionSphere, new THREE.Vector3());
            if (!hit) return null;
            planet.worldToLocal(hit).normalize();
            return {
                latitude: Math.asin(THREE.MathUtils.clamp(hit.y, -1, 1)) / DEG,
                longitude: wrapLongitude(Math.atan2(-hit.z, hit.x) / DEG + CONFIG.elevationMapCenterLon),
            };
        }

        function boundsBetween(start, end) {
            // Choose the shorter longitude interval, including across +/-180.
            const delta = wrapLongitude(end.longitude - start.longitude);
            const west = start.longitude + Math.min(0, delta);
            return {
                top_left: { latitude: Math.max(start.latitude, end.latitude), longitude: wrapLongitude(west) },
                bottom_right: { latitude: Math.min(start.latitude, end.latitude), longitude: wrapLongitude(west + Math.abs(delta)) },
                crosses_antimeridian: wrapLongitude(west) > wrapLongitude(west + Math.abs(delta)),
            };
        }

        function drawRegion(bounds) {
            while (roiLayer.children.length) {
                const object = roiLayer.children[0];
                roiLayer.remove(object);
                object.geometry.dispose();
                object.material.dispose();
            }
            if (!bounds) return;
            const north = bounds.top_left.latitude, south = bounds.bottom_right.latitude;
            const west = bounds.top_left.longitude;
            const east = bounds.bottom_right.longitude + (bounds.crosses_antimeridian ? 360 : 0);
            function point(lat, lon) {
                const elevation = sampleElevation(lat, lon);
                return latLonToVector3(lat, lon, CONFIG.radius + elevation * CONFIG.displacementScale
                    + CONFIG.displacementBias + 0.0015);
            }
            const vertices = [], indices = [], outline = [];
            const steps = 48;
            for (let row = 0; row <= steps; row++) {
                for (let col = 0; col <= steps; col++) {
                    const p = point(north + (south - north) * row / steps,
                                    west + (east - west) * col / steps);
                    vertices.push(p.x, p.y, p.z);
                    if (row < steps && col < steps) {
                        const i = row * (steps + 1) + col;
                        indices.push(i, i + steps + 1, i + 1, i + 1, i + steps + 1, i + steps + 2);
                    }
                }
            }
            for (let edge = 0; edge < 4; edge++) {
                for (let i = 0; i <= steps; i++) {
                    const t = i / steps;
                    outline.push(edge === 0 ? point(north, west + (east - west) * t)
                        : edge === 1 ? point(north + (south - north) * t, east)
                        : edge === 2 ? point(south, east + (west - east) * t)
                        : point(south + (north - south) * t, west));
                }
            }
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            geometry.setIndex(indices);
            roiLayer.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
                color: 0xfad6a5, transparent: true, opacity: 0.12,
                side: THREE.DoubleSide, depthWrite: false,
                polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
            })));
            roiLayer.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(outline),
                new THREE.LineBasicMaterial({color: 0xffdfad, depthWrite: false})));
        }

        function selectedSectors(bounds) {
            if (!bounds || bounds.crosses_antimeridian || !coverageData) return [];
            const ids = new Map();
            for (const patch of coverageData.rectangles) {
                if (patch.sector_id && patch.top_left[0] < bounds.bottom_right.longitude
                    && patch.bottom_right[0] > bounds.top_left.longitude
                    && patch.top_left[1] > bounds.bottom_right.latitude
                    && patch.bottom_right[1] < bounds.top_left.latitude) ids.set(patch.sector_id, patch);
            }
            return [...ids.values()];
        }
        $('region-confirm').addEventListener('click', () => {
            const matches = selectedSectors(region);
            if (matches.length !== 1) return;
            const url = new URL(matches[0].viewer_url, window.location.origin);
            url.search = new URLSearchParams({west: region.top_left.longitude,
                north: region.top_left.latitude, east: region.bottom_right.longitude,
                south: region.bottom_right.latitude});
            window.location.assign(url);
        });
        function showRegion(bounds, message) {
            $('region-status').hidden = !bounds;
            $('region-status').textContent = message;
            $('region-bounds').hidden = !bounds;
            $('region-clear').hidden = !bounds;
            const matches = selectedSectors(bounds);
            const confirm = $('region-confirm');
            confirm.hidden = !bounds;
            confirm.disabled = !!roiDrag || matches.length !== 1;
            if (bounds && !roiDrag) {
                $('region-status').textContent += bounds.crosses_antimeridian
                    ? ' Select on one side of the antimeridian.'
                    : matches.length > 1 ? ' Multiple sectors selected; narrow the selection.'
                    : matches.length === 0 ? ' No covered sector selected.' : '';
            }
            if (bounds) {
                const format = (p) => p.latitude.toFixed(4) + '\u00b0, ' + p.longitude.toFixed(4) + '\u00b0';
                $('region-top-left').textContent = format(bounds.top_left);
                $('region-bottom-right').textContent = format(bounds.bottom_right);
            }
        }

        function finishRegion(e, cancel = false) {
            if (!roiDrag || (e && e.pointerId !== roiDrag.pointerId)) return;
            const drag = roiDrag;
            const end = e && surfaceCoordinate(e);
            const moved = e && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) >= 5;
            let next = !cancel && moved && end ? boundsBetween(drag.start, end) : null;
            if (next && (next.top_left.latitude - next.bottom_right.latitude < 0.00001
                || Math.abs(wrapLongitude(next.bottom_right.longitude - next.top_left.longitude)) < 0.00001)) next = null;
            roiDrag = null;
            controls.enabled = true;
            if (canvas.hasPointerCapture(drag.pointerId)) canvas.releasePointerCapture(drag.pointerId);
            canvas.style.cursor = 'grab';
            if (next) {
                region = next;
                canvas.dataset.region = JSON.stringify(region);
                // A future view can consume these bounds without a server request today.
                canvas.dispatchEvent(new CustomEvent('regionselect', {detail: region, bubbles: true}));
            }
            drawRegion(region);
            showRegion(region, next
                ? (region.crosses_antimeridian ? 'Selected region crosses +/-180 degrees. Escape to clear.' : 'Region selected. Escape to clear.')
                : 'Selection cancelled. Right-drag on Mars to select a region.');
        }

        canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        canvas.addEventListener('pointerdown', (e) => {
            if (e.button !== 2) return;
            e.preventDefault();
            e.stopImmediatePropagation(); // Keep OrbitControls out of right-drag selection.
            if (roiDrag) return;
            const start = surfaceCoordinate(e);
            if (!start) return;
            controls.enabled = false;
            rotating = false;
            rotationSpeed = 0;
            renderToggle();
            select(null);
            roiDrag = {start, x: e.clientX, y: e.clientY, pointerId: e.pointerId};
            canvas.setPointerCapture(e.pointerId);
            canvas.style.cursor = 'crosshair';
            showRegion(region, 'Drag to set the opposite corner. Release on Mars to keep it.');
        }, true);
        canvas.addEventListener('pointermove', (e) => {
            if (!roiDrag || e.pointerId !== roiDrag.pointerId) return;
            e.stopImmediatePropagation();
            const end = surfaceCoordinate(e);
            if (!end) {
                showRegion(region, 'Outside Mars. Release here to cancel.');
                return;
            }
            const preview = boundsBetween(roiDrag.start, end);
            drawRegion(preview);
            showRegion(preview, 'Selecting region. Release to keep it.');
        }, true);
        canvas.addEventListener('pointerup', (e) => {
            if (!roiDrag || e.pointerId !== roiDrag.pointerId) return;
            e.preventDefault();
            e.stopImmediatePropagation();
            finishRegion(e);
        }, true);
        canvas.addEventListener('pointercancel', (e) => finishRegion(e, true));
        canvas.addEventListener('lostpointercapture', (e) => finishRegion(e, true));
        window.addEventListener('blur', () => finishRegion(null, true));
        function clearRegion() {
            finishRegion(null, true);
            region = null;
            delete canvas.dataset.region;
            drawRegion(null);
            showRegion(null, 'Right-drag on Mars to select a region.');
            canvas.dispatchEvent(new CustomEvent('regionselect', {detail: null, bubbles: true}));
        }
        $('region-clear').addEventListener('click', clearRegion);
        window.addEventListener('keydown', (e) => { if (e.key === 'Escape') clearRegion(); });

        // A click is a press and release without dragging, so rotating never selects a pin
        let pressStart = null;
        canvas.addEventListener('pointerdown', (e) => {
            if (e.button !== 0 || roiDrag) return;
            pressStart = { x: e.clientX, y: e.clientY };
            canvas.style.cursor = 'grabbing';
        });
        canvas.addEventListener('pointerup', (e) => {
            if (e.button !== 0) return;
            canvas.style.cursor = 'grab';
            if (!pressStart) return;
            const moved = Math.hypot(e.clientX - pressStart.x, e.clientY - pressStart.y);
            pressStart = null;
            if (moved < 5) select(findMarkerAt(e.clientX, e.clientY));
        });
        canvas.addEventListener('pointermove', (e) => {
            if (e.buttons) return;
            canvas.style.cursor = findMarkerAt(e.clientX, e.clientY) ? 'pointer' : 'grab';
        });
        $('label-close').addEventListener('click', () => select(null));
        window.addEventListener('keydown', (e) => { if (e.key === 'Escape') select(null); });
 
        // ---- Legend and display controls ------------------------------
        const legendRows = {};
        const typeInputs = {};
        const groupInputs = {};
        let masterInput = null;
 
        function setTypesVisible(ids, on) {
            ids.forEach((id) => {
                typeVisible[id] = on;
                markers.forEach((m) => { if (m.type.id === id) m.target = on ? 1 : 0; });
            });
            if (selected && !typeVisible[selected.type.id]) select(null);
            syncControls();
        }
 
        function syncControls() {
            types.forEach((t) => {
                const on = typeVisible[t.id];
                typeInputs[t.id].checked = on;
                typeInputs[t.id].closest('.row').classList.toggle('is-off', !on);
                legendRows[t.id].classList.toggle('is-off', !on);
            });
            const syncTri = (input, ids) => {
                const count = ids.filter((id) => typeVisible[id]).length;
                input.checked = count === ids.length;
                input.indeterminate = count > 0 && count < ids.length;
            };
            groups.forEach((g) => {
                if (groupInputs[g.id]) syncTri(groupInputs[g.id], types.filter((t) => t.group === g.id).map((t) => t.id));
            });
            syncTri(masterInput, types.map((t) => t.id));
        }
 
        // One switch row: a real checkbox, an optional swatch, a label, and the visual switch
        function switchRow({ label, color, swatch, extraClass }) {
            const input = el('input', { type: 'checkbox', checked: true });
            const row = el('label', { class: 'row ' + (extraClass || '') }, input);
            if (color) row.style.setProperty('--c', color);
            if (swatch) row.append(swatch);
            row.append(el('span', { class: 'row-label' }, label), el('span', { class: 'switch' }));
            return { row, input };
        }
 
        const legendBody = $('legend-body');
        const markerControls = $('marker-controls');
 
        const master = switchRow({ label: 'Markers', extraClass: 'section-head' });
        masterInput = master.input;
        master.input.addEventListener('change', () => setTypesVisible(types.map((t) => t.id), master.input.checked));
        markerControls.append(master.row);
 
        groups.forEach((g) => {
            const groupTypes = types.filter((t) => t.group === g.id);
            if (!groupTypes.length) return;
 
            // legend
            legendBody.append(el('p', { class: 'group-title' }, g.label));
            groupTypes.forEach((t) => {
                const row = el('div', { class: 'legend-row' }, createSwatch(t), el('span', { class: 'row-label' }, t.label));
                legendRows[t.id] = row;
                legendBody.append(row);
            });
 
            // controls
            const head = switchRow({ label: g.label, extraClass: 'row-group' });
            groupInputs[g.id] = head.input;
            head.input.addEventListener('change', () => setTypesVisible(groupTypes.map((t) => t.id), head.input.checked));
            markerControls.append(head.row);
 
            groupTypes.forEach((t) => {
                const item = switchRow({ label: t.label, color: t.color, swatch: createSwatch(t), extraClass: 'row-indent' });
                typeInputs[t.id] = item.input;
                item.input.addEventListener('change', () => setTypesVisible([t.id], item.input.checked));
                markerControls.append(item.row);
            });
        });
        syncControls();
 
        // Collapsible panels (start collapsed on small screens)
        const startOpen = window.innerWidth > 640;
        [$('legend'), $('controls')].forEach((panel) => {
            const button = panel.querySelector('.panel-head');
            const setOpen = (open) => {
                panel.dataset.open = String(open);
                button.setAttribute('aria-expanded', String(open));
            };
            button.addEventListener('click', () => setOpen(panel.dataset.open !== 'true'));
            setOpen(startOpen);
        });
 
        // ---- Resize ---------------------------------------------------
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
 
        // ---- Render loop -----------------------------------------------
        const clock = new THREE.Clock();
        function animate() {
            requestAnimationFrame(animate);
            const dt = Math.min(clock.getDelta(), 0.1);
 
            // Ease toward the target speed so pausing/resuming feels smooth
            const target = rotating ? CONFIG.rotationSpeed : 0;
            rotationSpeed += (target - rotationSpeed) * Math.min(1, dt * 3);
            planet.rotation.y += rotationSpeed * dt;
 
            // Pins grow in and shrink away when their group is switched on or off
            markers.forEach((m) => {
                if (m.vis === m.target) return;
                m.vis += (m.target - m.vis) * Math.min(1, dt * 12);
                if (Math.abs(m.target - m.vis) < 0.01) m.vis = m.target;
                m.group.visible = m.vis > 0.005;
                m.group.scale.setScalar(Math.max(m.vis, 0.0001));
            });
 
            // Gentler dragging when zoomed in
            const t = THREE.MathUtils.clamp(
                (camera.position.length() - controls.minDistance) / (controls.maxDistance - controls.minDistance), 0, 1);
            controls.rotateSpeed = 0.15 + 0.55 * t;
            if (!roiDrag) controls.update();
 
            renderer.render(scene, camera);
            updateLabel();
        }
        animate();
 
        loadingEl.classList.add('is-done');
    }
 
    window.addEventListener('load', () => {
        init().catch((err) => {
            console.error(err);
            $('loading').textContent =
                err.message + '. Check the server and that the texture files are available in static/texture/.';
        });
    });
})();
 
