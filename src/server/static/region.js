(() => {
    'use strict';
    const config = JSON.parse(document.getElementById('region-config').textContent);
    const canvas = document.getElementById('region-canvas');
    const context = canvas.getContext('2d');
    const status = document.getElementById('map-status');
    let view, frame = null, drag = null, timer, controller, generation = 0;
    const initial = config.initial_view;
    const photos = window.createPhotoStops(config, canvas);
    const endpoints = [];
    let route = [], routeController = null, routing = false, mode3D = false;
    const routeStatus = document.getElementById('route-status');
    const routeButton = document.getElementById('calculate-route');
    const clearButton = document.getElementById('clear-endpoints');
    const subdivisionSlider = document.getElementById('subdivisions');
    const mouseCoordinate = document.getElementById('mouse-coordinate');
    const constraintOptions = document.getElementById('constraint-options');
    const constraintStatus = document.getElementById('constraints-status');
    const profileSelect = document.getElementById('routing-profile');
    const profileButton = document.getElementById('profile-settings-button');
    const profilePanel = document.getElementById('profile-panel');
    // Older cached templates nest this inside the blurred routing panel,
    // which makes fixed positioning relative to that panel, not the viewport.
    const constraintsPanel = constraintOptions.closest('.constraints-panel');
    if (constraintsPanel.closest('.routing-panel')) document.body.append(constraintsPanel);
    let heatmaps = null;
    const terrain = window.createTerrainPreview?.(config, {
        getView: () => view,
        getRoute: () => route,
        getEndpoints: () => endpoints.map(point => [point.x, point.y]),
        getHeatmaps: () => heatmaps,
        onModeChange(active) {
            mode3D = active;
            document.body.classList.toggle('is-3d', active);
            const panel = document.querySelector('.routing-panel');
            panel.inert = active;
            panel.setAttribute('aria-disabled', String(active));
            if (profileButton) profileButton.disabled = active;
            if (profilePanel) profilePanel.hidden = true;
            if (active) {
                controller?.abort(); clearTimeout(timer); generation++; drag = null;
                document.getElementById('photo-stop')?.close();
            } else resize();
            updateRouteControls();
        },
    });

    if (profileButton && profilePanel && profileSelect) {
        function setProfilePanel(open) {
            profilePanel.hidden = !open;
            profileButton.setAttribute('aria-expanded', String(open));
            if (open) profileSelect.focus();
        }
        profileButton.addEventListener('click', () => setProfilePanel(profilePanel.hidden));
        document.addEventListener('pointerdown', (event) => {
            if (!profileButton.parentElement.contains(event.target)) setProfilePanel(false);
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && !profilePanel.hidden) {
                setProfilePanel(false); profileButton.focus();
            }
        });
        profileSelect.addEventListener('change', () => {
            route = []; clearHeatmaps(); routeStatus.textContent = '';
            const label = 'Routing profile: ' + profileSelect.selectedOptions[0].textContent;
            profileButton.title = label;
            profileButton.setAttribute('aria-label', label);
            draw();
        });
    }

    function clearHeatmaps() {
        heatmaps = null;
        constraintOptions.replaceChildren();
        constraintStatus.textContent = 'Available after routing.';
    }
    async function loadHeatmaps(data) {
        if (!data) return;
        const layers = await Promise.all(data.layers.map(async (layer, index) => {
            const image = new Image();
            image.src = layer.image;
            await image.decode();
            return {...layer, image, visible: index === 0};
        }));
        heatmaps = {...data, layers};
        constraintOptions.replaceChildren();
        constraintStatus.textContent = 'Magenta: blocked · gaps: no data';
        layers.forEach((layer) => {
            const row = document.createElement('div'); row.className = 'constraint-option';
            const label = document.createElement('label');
            const checkbox = document.createElement('input'); checkbox.type = 'checkbox';
            checkbox.checked = layer.visible;
            checkbox.addEventListener('change', () => { layer.visible = checkbox.checked; draw(); });
            const names = {
                'Slope penalty (maximum finite outgoing)': 'Slope cost',
                'Slope-blocked directions (fraction)': 'Blocked directions',
                'Combined cell cost': 'Cell cost',
            };
            const title = document.createElement('span');
            title.textContent = names[layer.title] || layer.title.replace(/\s*[—–-]\s*penalty$/i, '');
            label.title = layer.title;
            label.append(checkbox, title);
            const gradient = document.createElement('div'); gradient.className = 'constraint-gradient';
            gradient.style.background = 'linear-gradient(to right, ' + layer.colors.join(', ') + ')';
            const range = document.createElement('small');
            const number = (value) => new Intl.NumberFormat(undefined, {maximumSignificantDigits: 3}).format(value);
            range.textContent = layer.minimum === null ? 'No values' : number(layer.minimum) + ' – ' + number(layer.maximum);
            row.append(label, gradient, range); constraintOptions.append(row);
        });
    }
    function drawHeatmaps() {
        if (!heatmaps) return;
        const mesh = heatmaps.pixel_mesh;
        const nr = mesh.length - 1, nc = mesh[0].length - 1;
        const [rows, cols] = heatmaps.shape;
        const screen = (p) => [(p[0]-view.x)/view.width*canvas.width, (p[1]-view.y)/view.height*canvas.height];
        // Place native-resolution PNG cells through a small georeferencing
        // mesh. No numeric grid resampling or image smoothing is performed.
        context.save(); context.globalAlpha = .55; context.imageSmoothingEnabled = false;
        context.beginPath();
        context.rect(-view.x/view.width*canvas.width, -view.y/view.height*canvas.height,
                     config.width/view.width*canvas.width, config.height/view.height*canvas.height);
        context.clip();
        function triangle(image, source, target) {
            if (Math.max(...target.map(p => p[0])) < 0 || Math.min(...target.map(p => p[0])) > canvas.width
                || Math.max(...target.map(p => p[1])) < 0 || Math.min(...target.map(p => p[1])) > canvas.height) return;
            const [s0, s1, s2] = source, [p0, p1, p2] = target;
            const ux = s1[0]-s0[0], uy = s1[1]-s0[1], vx = s2[0]-s0[0], vy = s2[1]-s0[1];
            const det = ux*vy-uy*vx;
            const a = ((p1[0]-p0[0])*vy-(p2[0]-p0[0])*uy)/det;
            const c = ((p2[0]-p0[0])*ux-(p1[0]-p0[0])*vx)/det;
            const b = ((p1[1]-p0[1])*vy-(p2[1]-p0[1])*uy)/det;
            const d = ((p2[1]-p0[1])*ux-(p1[1]-p0[1])*vx)/det;
            context.save(); context.beginPath(); context.moveTo(...p0); context.lineTo(...p1); context.lineTo(...p2);
            context.closePath(); context.clip();
            context.transform(a, b, c, d, p0[0]-a*s0[0]-c*s0[1], p0[1]-b*s0[0]-d*s0[1]);
            context.drawImage(image, 0, 0); context.restore();
        }
        heatmaps.layers.filter(layer => layer.visible).forEach((layer) => {
            for (let r = 0; r < nr; r++) for (let c = 0; c < nc; c++) {
                const source = [[c*cols/nc, r*rows/nr], [(c+1)*cols/nc, r*rows/nr],
                                [(c+1)*cols/nc, (r+1)*rows/nr], [c*cols/nc, (r+1)*rows/nr]];
                const target = [mesh[r][c], mesh[r][c+1], mesh[r+1][c+1], mesh[r+1][c]].map(screen);
                for (const indices of [[0, 1, 2], [0, 2, 3]])
                    triangle(layer.image, indices.map(i => source[i]), indices.map(i => target[i]));
            }
        });
        context.restore();
    }

    function pixelAt(event) {
        const rect = canvas.getBoundingClientRect();
        const x = view.x + (event.clientX-rect.left)/rect.width*view.width;
        const y = view.y + (event.clientY-rect.top)/rect.height*view.height;
        return x >= 0 && y >= 0 && x < config.width && y < config.height ? {x, y} : null;
    }
    function geographic(point) {
        // A bounded metadata grid avoids a server request for every mouse move.
        // The server converts endpoint pixels with the exact CRS for routing.
        const grid = config.coordinate_grid;
        const x = point.x / config.width * (grid[0].length-1);
        const y = point.y / config.height * (grid.length-1);
        const c = Math.min(grid[0].length-2, Math.floor(x));
        const r = Math.min(grid.length-2, Math.floor(y));
        const tx = x-c, ty = y-r;
        return [0, 1].map((axis) =>
            grid[r][c][axis]*(1-tx)*(1-ty) + grid[r][c+1][axis]*tx*(1-ty)
            + grid[r+1][c][axis]*(1-tx)*ty + grid[r+1][c+1][axis]*tx*ty);
    }
    function formatPoint(point) {
        const [lon, lat] = geographic(point);
        return '(' + lat.toFixed(5) + ', ' + (((lon+180)%360+360)%360-180).toFixed(5) + ')';
    }
    function updateRouteControls() {
        routeButton.disabled = mode3D || routing || endpoints.length !== 2;
        clearButton.disabled = mode3D || routing || endpoints.length === 0;
        subdivisionSlider.disabled = mode3D || routing;
        if (profileSelect) profileSelect.disabled = mode3D || routing;
        const terrainButton = document.getElementById('terrain-toggle');
        if (terrainButton) terrainButton.disabled = routing;
        const list = document.getElementById('route-endpoints');
        list.replaceChildren(...['Start', 'End'].map((name, index) => {
            const item = document.createElement('li');
            item.textContent = name + ': ' + (endpoints[index] ? formatPoint(endpoints[index]) : '—');
            return item;
        }));
    }
    function updateFineness() {
        clearHeatmaps();
        const amount = Number(subdivisionSlider.value);
        document.getElementById('subdivision-value').textContent = amount;
        const columns = Math.min(config.width-1, amount)+1;
        const rows = Math.min(config.height-1, amount)+1;
        document.getElementById('grid-estimate').textContent =
            columns + ' × ' + rows + ' grid';
        route = [];
        routeStatus.textContent = '';
        if (view) draw();
    }
    function drawRouting() {
        const ratio = canvas.width / canvas.getBoundingClientRect().width;
        const screen = (p) => ({x: (p[0]-view.x)/view.width*canvas.width,
                                y: (p[1]-view.y)/view.height*canvas.height});
        context.save();
        if (route.length) {
            const points = route.map(screen);
            context.beginPath(); context.moveTo(points[0].x, points[0].y);
            // Round only a small portion of each corner; authoritative route
            // vertices remain unchanged in source pixels and geographic JSON.
            for (let i = 1; i < points.length-1; i++) {
                const a = points[i-1], b = points[i], c = points[i+1];
                const before = Math.hypot(b.x-a.x, b.y-a.y), after = Math.hypot(c.x-b.x, c.y-b.y);
                const radius = 0.12 * Math.min(before, after);
                if (!before || !after) continue;
                context.lineTo(b.x-(b.x-a.x)/before*radius, b.y-(b.y-a.y)/before*radius);
                context.quadraticCurveTo(b.x, b.y, b.x+(c.x-b.x)/after*radius, b.y+(c.y-b.y)/after*radius);
            }
            const last = points[points.length-1]; context.lineTo(last.x, last.y);
            context.lineJoin = 'round'; context.lineCap = 'round';
            context.lineWidth = 6*ratio; context.strokeStyle = '#18272d'; context.stroke();
            context.lineWidth = 3*ratio; context.strokeStyle = '#a4f5d5'; context.stroke();
        }
        endpoints.forEach((point, index) => {
            const p = screen([point.x, point.y]);
            context.beginPath(); context.arc(p.x, p.y, 7*ratio, 0, Math.PI*2);
            context.fillStyle = index ? '#ffc28a' : '#a4f5d5'; context.fill();
            context.lineWidth = 2*ratio; context.strokeStyle = '#17222a'; context.stroke();
            context.font = 'bold ' + 12*ratio + 'px system-ui';
            context.fillStyle = '#fff'; context.fillText(index ? 'B' : 'A', p.x+11*ratio, p.y-8*ratio);
        });
        context.restore();
    }
    canvas.addEventListener('contextmenu', (event) => event.preventDefault());
    canvas.addEventListener('pointerdown', (event) => {
        if (event.button !== 2) return;
        event.preventDefault();
        if (routing || mode3D) return;
        const point = pixelAt(event);
        if (!point) { routeStatus.textContent = 'Select a point inside the sector.'; return; }
        if (endpoints.length === 2) endpoints.length = 0;
        endpoints.push(point); route = []; clearHeatmaps();
        routeStatus.textContent = '';
        updateRouteControls(); draw();
    });
    canvas.addEventListener('pointermove', (event) => {
        const point = pixelAt(event);
        mouseCoordinate.textContent = point ? formatPoint(point) : 'Outside sector';
    });
    canvas.addEventListener('pointerleave', () => { mouseCoordinate.textContent = '-'; });
    clearButton.addEventListener('click', () => {
        if (mode3D) return;
        endpoints.length = 0; route = []; clearHeatmaps();
        routeStatus.textContent = '';
        updateRouteControls(); draw();
    });
    subdivisionSlider.addEventListener('input', updateFineness);
    routeButton.addEventListener('click', async () => {
        if (mode3D || routing || endpoints.length !== 2) return;
        routing = true; route = []; clearHeatmaps(); updateRouteControls(); draw();
        routeStatus.textContent = 'Computing route...';
        routeController = new AbortController();
        try {
            const response = await fetch(config.route_url, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, signal: routeController.signal,
                body: JSON.stringify({start: endpoints[0], end: endpoints[1], subdivisions: Number(subdivisionSlider.value),
                                      profile: {type: profileSelect?.value || 'human'}}),
            });
            const result = await response.json();
            try { await loadHeatmaps(result.heatmaps); }
            catch (error) { constraintStatus.textContent = 'Constraint previews could not be loaded.'; console.warn(error); }
            if (!response.ok) throw new Error(result.error || 'Routing failed.');
            if (!result.path.length) throw new Error('No route was found.');
            route = result.path;
            routeStatus.textContent = result.path.length === 1
                ? 'Both points snap to the same grid cell. Increase subdivisions for more detail.'
                : 'Route found on a ' + result.grid_shape[1] + ' x ' + result.grid_shape[0] + ' grid.';
        } catch (error) {
            if (error.name !== 'AbortError') routeStatus.textContent = error.message;
        } finally {
            routing = false; updateRouteControls(); draw();
        }
    });
    updateFineness(); updateRouteControls();

    function constrain() {
        // Leave a margin when a whole sector is smaller than the viewport.
        view.x = view.width >= config.width ? (config.width - view.width) / 2
            : Math.max(0, Math.min(config.width - view.width, view.x));
        view.y = view.height >= config.height ? (config.height - view.height) / 2
            : Math.max(0, Math.min(config.height - view.height, view.y));
    }
    function draw() {
        if (mode3D) { terrain?.updateOverlays(heatmaps); return; }
        photos.update(view);
        context.fillStyle = '#22272e';
        context.fillRect(0, 0, canvas.width, canvas.height);
        if (frame) {
            const bounds = frame.bounds;
            context.drawImage(frame.image,
            (bounds.x-view.x)/view.width*canvas.width,
            (bounds.y-view.y)/view.height*canvas.height,
            bounds.width/view.width*canvas.width, bounds.height/view.height*canvas.height);
        }
        drawHeatmaps();
        drawRouting();
    }
    function schedule() {
        generation++;
        controller?.abort();
        clearTimeout(timer);
        draw();
        timer = setTimeout(load, 150);
    }
    async function load() {
        const version = generation;
        controller = new AbortController();
        const signal = controller.signal;
        const url = new URL(config.frame_url, window.location.origin);
        url.search = new URLSearchParams({...view, pixels: 1024});
        status.textContent = 'Loading visible area...';
        let image;
        try {
            const response = await fetch(url, {signal});
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Could not load map');
            }
            const bounds = JSON.parse(response.headers.get('X-Viewport'));
            image = await createImageBitmap(await response.blob());
            if (version !== generation) { image.close(); return; }
            frame?.image.close();
            frame = {image, bounds};
            draw();
            status.textContent = 'Viewport loaded | ' + (config.width / view.width).toFixed(1)
                + 'x | ' + image.width + ' x ' + image.height + ' image';
        } catch (error) {
            if (error.name !== 'AbortError' && version === generation)
                status.textContent = error.message + '. Move the map or use Fit selection to retry.';
        }
    }
    function fit() {
        if (mode3D) return;
        const aspect = canvas.width / canvas.height;
        const width = Math.max(initial.width, initial.height * aspect) * 1.05;
        const height = width / aspect;
        view = {x: initial.x + initial.width/2 - width/2,
                y: initial.y + initial.height/2 - height/2, width, height};
        constrain(); schedule();
    }
    function resize() {
        if (mode3D) return;
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = Math.round(innerWidth * ratio);
        canvas.height = Math.round(innerHeight * ratio);
        if (!view) { fit(); return; }
        const centre = view.y + view.height / 2;
        view.height = view.width * canvas.height / canvas.width;
        view.y = centre - view.height / 2;
        constrain(); schedule();
    }
    function zoom(factor, px = 0.5, py = 0.5) {
        if (mode3D) return;
        const width = Math.max(8, Math.min(config.width * 2, view.width * factor));
        const height = width * canvas.height / canvas.width;
        view.x += (view.width-width)*px;
        view.y += (view.height-height)*py;
        view.width = width; view.height = height;
        constrain(); schedule();
    }
    canvas.addEventListener('wheel', (event) => {
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        zoom(Math.exp(Math.max(-0.8, Math.min(0.8, event.deltaY*0.003))),
             (event.clientX-rect.left)/rect.width, (event.clientY-rect.top)/rect.height);
    }, {passive: false});
    canvas.addEventListener('pointerdown', (event) => {
        if (mode3D || event.button !== 0) return;
        drag = {id: event.pointerId, x: event.clientX, y: event.clientY};
        canvas.setPointerCapture(event.pointerId); canvas.style.cursor = 'grabbing';
    });
    canvas.addEventListener('pointermove', (event) => {
        if (!drag || event.pointerId !== drag.id) return;
        const rect = canvas.getBoundingClientRect();
        view.x -= (event.clientX-drag.x)/rect.width*view.width;
        view.y -= (event.clientY-drag.y)/rect.height*view.height;
        drag.x = event.clientX; drag.y = event.clientY;
        constrain(); schedule();
        const point = pixelAt(event);
        mouseCoordinate.textContent = point ? formatPoint(point) : 'Outside sector';
    });
    function release() { drag = null; canvas.style.cursor = 'grab'; }
    canvas.addEventListener('pointerup', release);
    canvas.addEventListener('pointercancel', release);
    canvas.addEventListener('lostpointercapture', release);
    document.getElementById('zoom-in').addEventListener('click', () => zoom(0.5));
    document.getElementById('zoom-out').addEventListener('click', () => zoom(2));
    document.getElementById('fit-region').addEventListener('click', fit);
    window.addEventListener('resize', resize);
    window.addEventListener('pagehide', () => { controller?.abort(); clearTimeout(timer); frame?.image.close(); routeController?.abort(); terrain?.dispose(); });
    resize();
})();
