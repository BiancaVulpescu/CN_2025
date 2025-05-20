import React, { useState } from 'react';

const Tema8 = () => {
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleRun = async () => {
    setLoading(true);
    setOutput('');
    try {
      const res = await fetch('http://localhost:5000/tema8-output', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
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
      <h2>Minimizarea functiilor folosind metoda gradientului descendent (Tema 8)</h2>
      <button onClick={handleRun} disabled={loading} style={{marginBottom: '1em'}}>
        {loading ? 'Se execută...' : 'Rulează Tema 8'}
      </button>
      <pre style={{background:'#222', color:'#fff', padding:'1em', borderRadius:'8px', minHeight:'10em', maxHeight:'60vh', overflowY:'auto'}}>{output}</pre>
    </div>
  );
};

export default Tema8;