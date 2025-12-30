async function fetchState() {
  const res = await fetch('/api/brain/state');
  return await res.json();
}

async function setAttention(module, weight) {
  await fetch(`/api/brain/attention/${encodeURIComponent(module)}?weight=${weight}`, {
    method: 'POST'
  });
}

function render(state) {
  const tbody = document.querySelector('#attention tbody');
  tbody.innerHTML = '';
  for (const [name, weight] of Object.entries(state.attention)) {
    const tr = document.createElement('tr');
    const nameTd = document.createElement('td');
    nameTd.textContent = name;
    const weightTd = document.createElement('td');
    const input = document.createElement('input');
    input.type = 'range';
    input.min = 0;
    input.max = 1;
    input.step = 0.01;
    input.value = weight;
    input.addEventListener('input', () => setAttention(name, input.value));
    weightTd.appendChild(input);
    tr.appendChild(nameTd);
    tr.appendChild(weightTd);
    tbody.appendChild(tr);
  }
  document.getElementById('memoryHits').textContent = state.memory_hits;
}

async function poll() {
  const state = await fetchState();
  render(state);
}

setInterval(poll, 1000);
poll();
