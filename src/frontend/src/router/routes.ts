import LoginView from '../views/LoginView.vue';
import HomeView from '../views/HomeView.vue';
import DetailView from '../views/DetailView.vue';
import UserView from '../views/UserView.vue';
import MetricsView from '../views/MetricsView.vue';

export const routes = [
  { path: '/login', component: LoginView },
  { path: '/', component: HomeView },
  { path: '/detail', component: DetailView },
  { path: '/user', component: UserView },
  { path: '/metrics', component: MetricsView },
];
