import { useState } from 'react';

const defaultA = [
  [4.0, 0.0, 4.0],
  [1.0, 4.0, 2.0],
  [2.0, 4.0, 6.0],
];
const defaultdU = [2.0, 4.0, 1.0];
const defaultb = [6.0, 5.0, 12.0];

const Tema2 = () => {
  const [tip, setTip] = useState('mic');
  const [n, setN] = useState(3);
  const [epsilon, setEpsilon] = useState(1e-15);
  const [A, setA] = useState(JSON.stringify(defaultA));
  const [dU, setdU] = useState(JSON.stringify(defaultdU));
  const [b, setB] = useState(JSON.stringify(defaultb));
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setOutput('');
    let payload = { tip };
    if (tip === 'random') {
      payload.n = Number(n);
      payload.epsilon = Number(epsilon);
    } else {
      payload.n = 3;
      payload.epsilon = Number(epsilon);
      try {
        payload.A = JSON.parse(A);
        payload.dU = JSON.parse(dU);
        payload.b = JSON.parse(b);
      } catch (err) {
        setOutput('Eroare la parsarea matricelor/vectorilor. Verifica formatul JSON!');
        setLoading(false);
        return;
      }
    }
    try {
      const res = await fetch('http://localhost:5000/tema2-output', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setOutput(data.output);
    } catch (err) {
      setOutput('Eroare la conectarea cu serverul.');
    }
    setLoading(false);
  };

  return (
    <div>
      <h2>Descompunere LU - tema 2</h2>
      <form onSubmit={handleSubmit} style={{marginBottom: '1em'}}>
        <label>
          Tip execuție:
          <select value={tip} onChange={e => setTip(e.target.value)} style={{marginLeft: '1em'}}>
            <option value="mic">Exemplu mic</option>
            <option value="random">Random</option>
          </select>
        </label>
        {tip === 'random' && (
          <>
            <div style={{marginTop: '1em'}}>
              <label>Dimensiune matrice n: <input type="number" min="2" value={n} onChange={e => setN(e.target.value)} /></label>
            </div>
            <div style={{marginTop: '1em'}}>
              <label>Epsilon: <input type="number" step="any" value={epsilon} onChange={e => setEpsilon(e.target.value)} /></label>
            </div>
          </>
        )}
        {tip === 'mic' && (
          <>
            <div style={{marginTop: '1em'}}>
              <label>Matrice A (JSON): <input type="text" value={A} onChange={e => setA(e.target.value)} size={40} /></label>
            </div>
            <div style={{marginTop: '1em'}}>
              <label>Diagonala U (JSON): <input type="text" value={dU} onChange={e => setdU(e.target.value)} size={20} /></label>
            </div>
            <div style={{marginTop: '1em'}}>
              <label>Vector b (JSON): <input type="text" value={b} onChange={e => setB(e.target.value)} size={20} /></label>
            </div>
            <div style={{marginTop: '1em'}}>
              <label>Epsilon: <input type="number" step="any" value={epsilon} onChange={e => setEpsilon(e.target.value)} /></label>
            </div>
          </>
        )}
        <div style={{marginTop: '1em'}}>
          <button type="submit" disabled={loading}>{loading ? 'Se execută...' : 'Execută'}</button>
        </div>
      </form>
      <pre style={{background:'#222', color:'#fff', padding:'1em', borderRadius:'8px', minHeight:'10em'}}>{output}</pre>
    </div>
  );
};

export default Tema2;