/* Bounded, read-only terrain preview. Three.js is loaded only on first use. */
(() => {
    let libraries;
    function loadScript(url) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = url;
            script.onload = resolve;
            script.onerror = () => { script.remove(); reject(new Error('Could not load the 3D viewer.')); };
            document.head.append(script);
        });
    }
    function loadLibraries() {
        if (!libraries) {
            libraries = (async () => {
                if (!window.THREE) await loadScript('https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js');
                if (!window.THREE.OrbitControls)
                    await loadScript('https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js');
            })();
            libraries.catch(() => { libraries = null; });
        }
        return libraries;
    }
    const bytes = (encoded) => Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
    const HEIGHT_SCALE = 2;

    window.createTerrainPreview = function(config, hooks) {
        const container = document.getElementById('terrain-view');
        const button = document.getElementById('terrain-toggle');
        const slider = document.getElementById('mesh-detail');
        const output = document.getElementById('mesh-detail-value');
        const status = document.getElementById('terrain-status');
        if (!container || !button || !config.terrain_url) return {updateOverlays() {}, dispose() {}};
        let active = false, version = 0, request, timer, viewport;
        let renderer, scene, camera, controls, model, mesh, data, textureCanvas;
        let overlayData, overlayVisibility;

        function message(text) { status.textContent = text; status.hidden = !text; }
        function render() { if (active && renderer && model) renderer.render(scene, camera); }
        function resize() {
            if (!renderer) return;
            renderer.setSize(window.innerWidth, window.innerHeight, false);
            camera.aspect = window.innerWidth/window.innerHeight;
            model?.traverse(object => {
                object.material?.uniforms?.resolution?.value.set(window.innerWidth, window.innerHeight);
                if (object.isSprite && object.userData.endpoint) scalePin(object);
            });
            camera.updateProjectionMatrix(); render();
        }
        function initialize() {
            if (renderer) return;
            const T = window.THREE;
            renderer = new T.WebGLRenderer({antialias: true});
            renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
            renderer.setClearColor(0x181c22);
            renderer.outputEncoding = T.sRGBEncoding;
            renderer.domElement.setAttribute('aria-label', '3D terrain. Drag to orbit; scroll to zoom.');
            container.append(renderer.domElement);
            scene = new T.Scene();
            camera = new T.PerspectiveCamera(45, 1, .1, 1e7);
            controls = new T.OrbitControls(camera, renderer.domElement);
            controls.maxPolarAngle = Math.PI/2-.04;
            controls.screenSpacePanning = false;
            controls.zoomSpeed = 1.5;
            controls.addEventListener('change', render);
            scene.add(new T.HemisphereLight(0xf7f3ea, 0x5b5144, .9));
            const light = new T.DirectionalLight(0xfff4df, 1.1);
            light.position.set(-1, 2, 1); scene.add(light);
            resize();
        }
        function clearModel() {
            if (!model) return;
            scene.remove(model);
            const geometries = new Set();
            model.traverse(object => {
                // Sprite quads are shared internally by Three.js.
                if (object.geometry && !object.isSprite && !geometries.has(object.geometry)) {
                    geometries.add(object.geometry); object.geometry.dispose();
                }
                if (object.material) {
                    object.material.map?.dispose();
                    object.material.dispose();
                }
            });
            model = mesh = data = textureCanvas = null;
            overlayData = null;
        }
        function updateOverlays(heatmaps, force = false) {
            if (!mesh) return;
            const visibility = heatmaps?.layers.map(layer => layer.visible ? '1' : '0').join('') || '';
            if (!force && heatmaps === overlayData && visibility === overlayVisibility) return;
            overlayData = heatmaps; overlayVisibility = visibility;
            const ctx = textureCanvas.getContext('2d');
            ctx.fillStyle = '#c4b9a4'; ctx.fillRect(0, 0, textureCanvas.width, textureCanvas.height);
            window.drawConstraintLayers(ctx, heatmaps, data.viewport, textureCanvas.width, textureCanvas.height,
                                        config.width, config.height);
            mesh.material.map.needsUpdate = true;
            render();
        }
        function terrainSampler(positions, valid, rows, cols, bounds, span) {
            const stepX = bounds.width/cols, stepY = bounds.height/rows;
            return function sample(p) {
                if (p[0] < bounds.x || p[0] > bounds.x+bounds.width
                    || p[1] < bounds.y || p[1] > bounds.y+bounds.height) return null;
                const x = Math.max(0, Math.min(cols-1, (p[0]-bounds.x)/stepX-.5));
                const y = Math.max(0, Math.min(rows-1, (p[1]-bounds.y)/stepY-.5));
                const c = Math.min(cols-2, Math.floor(x)), r = Math.min(rows-2, Math.floor(y));
                const tx = x-c, ty = y-r, a = r*cols+c;
                // Match the mesh triangles rather than smoothing across holes.
                const nodes = tx+ty <= 1 ? [a, a+1, a+cols] : [a+cols+1, a+cols, a+1];
                const weights = tx+ty <= 1 ? [1-tx-ty, tx, ty] : [tx+ty-1, 1-tx, 1-ty];
                if (nodes.some(index => !valid[index])) return null;
                const result = [0, 1, 2].map(axis => nodes.reduce((sum, node, i) => sum+positions[node*3+axis]*weights[i], 0));
                result[1] += span*.0012;
                return result;
            };
        }
        function scalePin(sprite) {
            const height = 2*Math.tan(camera.fov*Math.PI/360)*48/window.innerHeight;
            sprite.scale.set(height*.75, height, 1);
        }
        function endpointPin(label, point) {
            const T = window.THREE;
            const canvas = document.createElement('canvas');
            canvas.width = 96; canvas.height = 128;
            const ctx = canvas.getContext('2d');
            ctx.beginPath(); ctx.moveTo(48, 122);
            ctx.bezierCurveTo(39, 102, 15, 75, 15, 47);
            ctx.bezierCurveTo(15, 3, 81, 3, 81, 47);
            ctx.bezierCurveTo(81, 75, 57, 102, 48, 122);
            ctx.closePath();
            ctx.fillStyle = label === 'A' ? '#a4f5d5' : '#ffc28a';
            ctx.fill(); ctx.strokeStyle = '#17292b'; ctx.lineWidth = 5; ctx.stroke();
            ctx.fillStyle = '#17292b'; ctx.font = '600 40px system-ui, sans-serif';
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(label, 48, 47);
            const texture = new T.CanvasTexture(canvas);
            texture.encoding = T.sRGBEncoding;
            const sprite = new T.Sprite(new T.SpriteMaterial({map: texture, transparent: true,
                sizeAttenuation: false, depthWrite: false, toneMapped: false}));
            sprite.name = 'route-endpoint-' + label;
            sprite.userData.endpoint = true;
            sprite.position.set(...point); sprite.center.set(.5, 0);
            sprite.renderOrder = 12;
            scalePin(sprite);
            return sprite;
        }
        function routeGeometry(positions, valid, rows, cols, bounds, span) {
            const points = hooks.getRoute();
            if (!points.length) return null;
            const output = [], stepX = bounds.width/cols, stepY = bounds.height/rows;
            const sample = terrainSampler(positions, valid, rows, cols, bounds, span);
            const limits = [bounds.x+stepX/2, bounds.y+stepY/2,
                            bounds.x+bounds.width-stepX/2, bounds.y+bounds.height-stepY/2];
            function clip(a, b) {
                let low = 0, high = 1;
                for (let axis = 0; axis < 2; axis++) {
                    const delta = b[axis]-a[axis];
                    if (!delta) { if (a[axis] < limits[axis] || a[axis] > limits[axis+2]) return null; }
                    else {
                        const t1 = (limits[axis]-a[axis])/delta, t2 = (limits[axis+2]-a[axis])/delta;
                        low = Math.max(low, Math.min(t1, t2)); high = Math.min(high, Math.max(t1, t2));
                    }
                }
                if (low > high) return null;
                return [low, high].map(t => [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t]);
            }
            for (let i = 1; i < points.length; i++) {
                const segment = clip(points[i-1], points[i]);
                if (!segment) continue;
                const [a, b] = segment;
                const divisions = Math.max(1, Math.min(4096, Math.ceil(2*Math.max(Math.abs(b[0]-a[0])/stepX, Math.abs(b[1]-a[1])/stepY))));
                let previous = sample(a);
                for (let j = 1; j <= divisions; j++) {
                    const next = sample([a[0]+(b[0]-a[0])*j/divisions, a[1]+(b[1]-a[1])*j/divisions]);
                    if (previous && next) output.push(...previous, ...next);
                    previous = next;
                }
            }
            if (!output.length) return null;
            const T = window.THREE;
            // Screen-space ribbons keep their width while orbiting/zooming.
            // Native WebGL lines are often limited to a single pixel.
            const vertices = [], starts = [], ends = [], sides = [];
            for (let i = 0; i < output.length; i += 6) {
                const a = output.slice(i, i+3), b = output.slice(i+3, i+6);
                for (const [end, side] of [[0, -1], [1, -1], [1, 1], [0, -1], [1, 1], [0, 1]]) {
                    vertices.push(...(end ? b : a)); starts.push(...a); ends.push(...b); sides.push(side);
                }
            }
            const geometry = new T.BufferGeometry();
            geometry.setAttribute('position', new T.Float32BufferAttribute(vertices, 3));
            geometry.setAttribute('segmentStart', new T.Float32BufferAttribute(starts, 3));
            geometry.setAttribute('segmentEnd', new T.Float32BufferAttribute(ends, 3));
            geometry.setAttribute('side', new T.Float32BufferAttribute(sides, 1));
            const group = new T.Group();
            for (const [width, color, order] of [[8, 0x17292b, 10], [4, 0xb2ffe0, 11]]) {
                const material = new T.ShaderMaterial({
                    uniforms: {resolution: {value: new T.Vector2(window.innerWidth, window.innerHeight)},
                               width: {value: width}, color: {value: new T.Color(color)}},
                    vertexShader: `
                        attribute vec3 segmentStart;
                        attribute vec3 segmentEnd;
                        attribute float side;
                        uniform vec2 resolution;
                        uniform float width;
                        void main() {
                            vec4 a = projectionMatrix * modelViewMatrix * vec4(segmentStart, 1.0);
                            vec4 b = projectionMatrix * modelViewMatrix * vec4(segmentEnd, 1.0);
                            vec2 direction = (b.xy / b.w - a.xy / a.w) * resolution;
                            vec2 normal = vec2(-direction.y, direction.x) / max(length(direction), 0.0001);
                            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                            gl_Position.xy += normal * width / resolution * side * gl_Position.w;
                        }`,
                    fragmentShader: 'uniform vec3 color; void main() { gl_FragColor = vec4(color, 1.0); }',
                    depthWrite: false, side: T.DoubleSide,
                });
                const ribbon = new T.Mesh(geometry, material);
                ribbon.renderOrder = order;
                ribbon.frustumCulled = false;
                group.add(ribbon);
            }
            return group;
        }
        function build(result, fitCamera) {
            const T = window.THREE, [rows, cols] = result.shape;
            const positions = new Float32Array(bytes(result.positions).buffer);
            const valid = bytes(result.valid);
            if (positions.length !== rows*cols*3 || valid.length !== rows*cols) throw new Error('Invalid terrain response.');
            // Display exaggeration only; source elevations and routing stay intact.
            for (let i = 1; i < positions.length; i += 3) positions[i] *= HEIGHT_SCALE;
            const indices = new Uint32Array((rows-1)*(cols-1)*6);
            let count = 0;
            for (let r = 0; r < rows-1; r++) for (let c = 0; c < cols-1; c++) {
                const a = r*cols+c, b = a+1, d = a+cols+1, e = a+cols;
                if (valid[a] && valid[e] && valid[b]) {indices[count++] = a; indices[count++] = e; indices[count++] = b;}
                if (valid[b] && valid[e] && valid[d]) {indices[count++] = b; indices[count++] = e; indices[count++] = d;}
            }
            if (!count) throw new Error('No continuous terrain here. Try a larger area or finer mesh.');
            const geometry = new T.BufferGeometry();
            geometry.setAttribute('position', new T.BufferAttribute(positions, 3));
            geometry.setIndex(new T.BufferAttribute(indices.subarray(0, count), 1));
            const uv = new Float32Array(rows*cols*2);
            for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
                uv[(r*cols+c)*2] = (c+.5)/cols; uv[(r*cols+c)*2+1] = 1-(r+.5)/rows;
            }
            geometry.setAttribute('uv', new T.BufferAttribute(uv, 2));
            geometry.computeVertexNormals(); geometry.computeBoundingBox();
            const size = geometry.boundingBox.getSize(new T.Vector3());
            const centre = geometry.boundingBox.getCenter(new T.Vector3());
            const span = Math.max(size.x, size.z, size.y, 1);
            clearModel(); data = result;
            model = new T.Group();
            textureCanvas = document.createElement('canvas');
            const ratio = Math.min(1024/result.viewport.width, 1024/result.viewport.height);
            textureCanvas.width = Math.max(1, Math.round(result.viewport.width*ratio));
            textureCanvas.height = Math.max(1, Math.round(result.viewport.height*ratio));
            const texture = new T.CanvasTexture(textureCanvas);
            texture.encoding = T.sRGBEncoding;
            texture.minFilter = texture.magFilter = T.NearestFilter;
            texture.generateMipmaps = false;
            mesh = new T.Mesh(geometry, new T.MeshStandardMaterial({map: texture, roughness: 1, side: T.DoubleSide}));
            model.add(mesh);
            const route = routeGeometry(positions, valid, rows, cols, result.viewport, span);
            if (route) model.add(route);
            const path = hooks.getRoute();
            const endpoints = path.length ? [path[0], path[path.length-1]] : (hooks.getEndpoints?.() || []);
            const sample = terrainSampler(positions, valid, rows, cols, result.viewport, span);
            endpoints.slice(0, 2).forEach((point, index) => {
                const position = sample(point);
                if (position) model.add(endpointPin(index ? 'B' : 'A', position));
            });
            scene.add(model);
            if (fitCamera) {
                const vertical = camera.fov*Math.PI/180;
                const horizontal = 2*Math.atan(Math.tan(vertical/2)*camera.aspect);
                const distance = size.length()/2/Math.sin(Math.min(vertical, horizontal)/2)*1.25;
                camera.near = Math.max(.01, distance/10000); camera.far = Math.max(100, distance*100);
                camera.updateProjectionMatrix();
                camera.position.copy(centre).add(new T.Vector3(.45, .65, .65).normalize().multiplyScalar(distance));
                controls.target.copy(centre);
                controls.minDistance = span*.03; controls.maxDistance = span*15;
            }
            controls.enabled = true; controls.update();
            updateOverlays(hooks.getHeatmaps(), true); render();
        }
        async function load(fitCamera = false) {
            const current = ++version;
            request?.abort(); request = new AbortController();
            message('Loading 3D terrain…');
            try {
                const url = new URL(config.terrain_url, window.location.origin);
                url.search = new URLSearchParams({...viewport, samples: slider.value});
                const response = await fetch(url, {signal: request.signal});
                const result = await response.json();
                if (!active || current !== version) return;
                if (!response.ok) throw new Error(result.error || 'Could not load terrain.');
                build(result, fitCamera);
                message('Read-only · ' + result.shape[1] + ' × ' + result.shape[0] + ' · height ×' + HEIGHT_SCALE);
            } catch (error) {
                if (error.name === 'AbortError' || current !== version) return;
                if (!model) hide();
                message(error.message);
            }
        }
        function hide() {
            active = false; version++; request?.abort(); clearTimeout(timer);
            if (controls) controls.enabled = false;
            clearModel(); container.hidden = true;
            button.setAttribute('aria-pressed', 'false');
            button.title = 'Show 3D terrain'; button.setAttribute('aria-label', button.title);
            message(''); hooks.onModeChange(false);
        }
        button.addEventListener('click', async () => {
            if (active) {hide(); return;}
            viewport = {...hooks.getView()};
            active = true; container.hidden = false;
            button.setAttribute('aria-pressed', 'true');
            button.title = 'Return to 2D map'; button.setAttribute('aria-label', button.title);
            hooks.onModeChange(true); message('Starting 3D preview…');
            const current = ++version;
            try {
                await loadLibraries();
                if (!active || version !== current) return;
                initialize(); resize(); await load(true);
            } catch (error) {
                if (!active || version !== current) return;
                hide(); message(error.message || '3D graphics are unavailable.');
            }
        });
        slider.addEventListener('input', () => {
            output.textContent = slider.value;
            clearTimeout(timer);
            if (active && renderer) timer = setTimeout(() => load(!model), 200);
        });
        window.addEventListener('resize', resize);
        return {updateOverlays, dispose() {
            active = false; version++; request?.abort(); clearTimeout(timer);
            clearModel(); controls?.dispose(); renderer?.dispose();
        }};
    };
})();
