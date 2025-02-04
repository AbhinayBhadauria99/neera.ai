import "./App.css";
import Header from "./components/Header";
import Section from "./components/Section";
import Input from "./components/Input";
import BGImage from "./assets/images/hero-desktop.jpg";

function App() {
  const screenSize = window.matchMedia("(min-width: 1024px)");
  let desktopView = screenSize.matches;
  return (
    <div className="app">
      <div className="lg-container">
        <Header />
        <Section />
        <Input />
      </div>
      {desktopView ? (
        <div className="img-container">
          <img src={BGImage} alt="Hero" />
        </div>
      ) : (
        ""
      )}
    </div>
  );
}

export default App;
