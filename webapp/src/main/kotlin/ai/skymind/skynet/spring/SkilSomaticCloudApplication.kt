package ai.skymind.skynet.spring

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication
import org.springframework.context.annotation.Bean
import org.springframework.security.config.annotation.web.builders.HttpSecurity
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter
import org.springframework.security.config.http.SessionCreationPolicy
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder

@SpringBootApplication
class SkilSomaticCloudApplication {
	@Bean
	fun webSecurityConfig(): WebSecurityConfigurerAdapter {
		return object : WebSecurityConfigurerAdapter() {
			override fun configure(http: HttpSecurity?) {
				http!!
						.csrf().disable()
						.authorizeRequests()
						.antMatchers("/**").permitAll().and()
						.sessionManagement().sessionCreationPolicy(SessionCreationPolicy.STATELESS)
			}
		}
	}

	@Bean
	fun passwordEncoder() = BCryptPasswordEncoder()
}

fun main(args: Array<String>) {
	runApplication<SkilSomaticCloudApplication>(*args)
}
