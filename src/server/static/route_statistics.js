/* Route measurements share a distance axis; no extra request when inspecting. */
window.createRouteStatistics = function() {
    const get = id => document.getElementById(id);
    const dialog = get('route-statistics'), button = get('route-profile-button');
    if (!dialog || !button) return {set() {}, clear() {}, open() {}, close() {}};
    const chart = get('route-chart'), selector = get('route-measurement');
    let samples = null;
    const number = value => new Intl.NumberFormat(undefined, {maximumFractionDigits: 1}).format(value);
    function svg(tag, attributes, text) {
        const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
        for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
        if (text !== undefined) node.textContent = text;
        chart.append(node); return node;
    }
    function close() { if (dialog.open) dialog.close(); }
    function draw() {
        if (!samples) return;
        const series = samples.series.find(item => item.id === selector.value) || samples.series[0];
        const distances = samples.distance_m, values = series.values;
        const finite = values.filter(value => Number.isFinite(value));
        get('route-statistics-title').textContent = series.label;
        chart.replaceChildren();
        const width = Math.max(280, Math.min(640, window.innerWidth-64)), height = 240;
        chart.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
        chart.setAttribute('aria-label', series.label + ' (' + series.unit + ') over horizontal distance along the route');
        const summary = get('route-chart-summary');
        if (!finite.length) { summary.textContent = 'No measurements available.'; return; }
        let low = Infinity, high = -Infinity;
        finite.forEach(value => {low = Math.min(low, value); high = Math.max(high, value);});
        const total = distances[distances.length-1];
        const distanceUnit = total >= 1000 ? 'km' : 'm', divisor = total >= 1000 ? 1000 : 1;
        summary.textContent = number(total/divisor) + ' ' + distanceUnit + ' · Min ' + number(low)
            + ' ' + series.unit + ' · Max ' + number(high) + ' ' + series.unit;
        const margin = Math.max((high-low)*.08, high === low ? Math.max(1, Math.abs(low)*.001) : .01);
        const bottom = 195, left = 66, right = width-18, top = 20;
        const y = value => bottom-(value-low+margin)/(high-low+2*margin)*(bottom-top);
        const x = distance => left+(total ? distance/total : 0)*(right-left);
        for (let i = 0; i <= 4; i++) {
            const value = low-margin+(high-low+2*margin)*i/4;
            svg('line', {x1: left, x2: right, y1: y(value), y2: y(value), class: 'chart-grid'});
            svg('text', {x: left-8, y: y(value)+4, 'text-anchor': 'end'}, number(value));
            if (total || i === 0) {
                const distance = total*i/4;
                svg('text', {x: x(distance), y: bottom+20, 'text-anchor': 'middle'}, number(distance/divisor));
            }
        }
        svg('text', {x: left, y: 12}, series.unit);
        svg('text', {x: (left+right)/2, y: height-3, 'text-anchor': 'middle'}, 'Distance (' + distanceUnit + ')');
        let path = '', connected = false;
        values.forEach((value, index) => {
            if (!Number.isFinite(value)) {connected = false; return;}
            path += (connected ? ' L ' : ' M ') + x(distances[index]) + ' ' + y(value);
            connected = true;
        });
        svg('path', {d: path, class: 'chart-measurement'});
        if (finite.length === 1) {
            const index = values.findIndex(value => Number.isFinite(value));
            svg('circle', {cx: x(distances[index]), cy: y(values[index]), r: 3, class: 'chart-point'});
        }
    }
    function open() {
        if (!samples) return;
        draw(); if (!dialog.open) dialog.show();
    }
    button.addEventListener('click', open);
    get('route-statistics-close').addEventListener('click', close);
    selector.addEventListener('change', draw);
    document.addEventListener('keydown', event => {if (event.key === 'Escape') close();});
    window.addEventListener('resize', () => {if (dialog.open) draw();});
    return {
        open, close,
        clear() {samples = null; button.hidden = true; close();},
        set(result) {
            close();
            const data = result.path_samples, count = result.path.length;
            if (!data || data.distance_m.length !== count || !count
                || !data.distance_m.every(value => Number.isFinite(value) && value >= 0)) {
                samples = null; button.hidden = true; return;
            }
            const series = data.series.filter(item => item.values.length === count);
            if (!series.length) {samples = null; button.hidden = true; return;}
            samples = {...data, series};
            selector.replaceChildren(...series.map(item => {
                const option = document.createElement('option'); option.value = item.id; option.textContent = item.label;
                return option;
            }));
            selector.value = series[0].id; selector.hidden = series.length < 2;
            button.hidden = false;
        },
    };
};
