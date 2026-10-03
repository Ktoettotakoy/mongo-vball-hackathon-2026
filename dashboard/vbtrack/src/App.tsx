import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Dashboard } from './views/Dashboard'
import { LandingPage } from './views/LandingPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
